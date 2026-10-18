#!/usr/bin/env python3
"""Guarded successor to frozen kernel 35; local diagnostic preparation only.

Weights, valid-input preprocessing and average-tie ranking are unchanged. Every
study and raw probability must pass validation before any submission is written.
Failures abort the run; DICOM recoveries and fp16-to-fp32 retries are recorded.
Three equal own-fold ranks form one component, blended 0.5/0.5 with public ranks.
No gold, CV or leaderboard result selects a weight. Models run sequentially.

Preprocessing and the public-model path originate in Dread Development's Apache-2.0
V1 scriptVersionId 344173275, already frozen and attributed by this project.
"""
import os, sys, glob, time, json, gc, hashlib
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
# T4 (Turing) cuDNN v9 has fp16/fp32 conv engines but NOT bf16 for these shapes
# ("GET was unable to find an engine..."); benchmark lets it pick a valid algo for
# the fixed (1,24,3,res,res) input.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ---- fixed config (must match training exactly) -----------------------------
IMG = 336
CROP_MM = 140.0
# 64 slices per study instead of 44, same proportions. Must match the corpus the weights
# were trained on (knee_corpus_v4.py).
SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
         ("Coronal", 0, 8), ("Axial", -1, 12)]
MAXS = sum(s[2] for s in SLOTS)                     # 64
K_EVAL = 42   # every window position the volume holds, not an evenly spaced subset
NORM = "imagenet"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# All three frozen own folds plus the fixed public component.
DRY_RUN = False
ARMS = [
    {"file": "fold0_selected_epoch7_model.pt", "kind": "own", "fold": 0,
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384},
    {"file": "fold1_selected_epoch6_model.pt", "kind": "own", "fold": 1,
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384},
    {"file": "fold2_selected_epoch8_model.pt", "kind": "own", "fold": 2,
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384},
    {"file": "raptor_ft_coatnet_v4_full.pt", "kind": "public", "fold": None,
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384},
]
EXPECTED_WEIGHT_SHA256 = {
    "fold0_selected_epoch7_model.pt": "aef1602b8bdf012c0e5f34204127bf1fe361e6c600435aa8e3224179231e9e0b",
    "fold1_selected_epoch6_model.pt": "e5038025a446f668d27421be69a868311d22574f7ea98c879878d2dfde53f8e5",
    "fold2_selected_epoch8_model.pt": "045e1557b508d367f648e8c0a984d9f0231bdb87ecb697d783e6733243b14ad2",
    "raptor_ft_coatnet_v4_full.pt": "89606f05849838529e1b4658d28fb049623205d9853371548403bca460361ded",
}


# ============================================================================
# Model -- verbatim from finetune_raptor.py
# ============================================================================
def build_backbone(arch, pretrained=False):
    # maxvit/maxxvit/coatnet are conv-attention hybrids: NO CLS token, NO interpolatable
    # pos-embed -> avg pool. The "vit" substring in "coatnet"/"maxvit" must NOT route them
    # down the ViT path (mirrors finetune_raptor.py exactly).
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if is_vit:
        kw.update(global_pool="token", dynamic_img_size=True)
    else:
        kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)


class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))


def load_model(pt_path, arch_default, res_default, device, ngpu=1):
    ck = torch.load(pt_path, map_location="cpu", weights_only=True)
    arch = ck.get("arch", ck.get("architecture", arch_default))
    ck_res = int(ck.get("res", res_default))
    bb = build_backbone(arch, pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    if "model" in ck:  # public schema
        state = ck["model"]
    elif "model_state_dict" in ck:  # honest-fold schema -> canonical public names
        state = {}
        for key, value in ck["model_state_dict"].items():
            key = key.replace("attention.", "att.")
            key = key.replace("classifier_weight", "clsW")
            key = key.replace("classifier_bias", "clsb")
            state[key] = value
    else:
        raise RuntimeError(f"unsupported checkpoint schema: {os.path.basename(pt_path)}")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    # NOTE: DataParallel removed on purpose. On the full hidden test it drove a system-RAM OOM
    # (per-forward module replication over many studies); a single T4 handles K_EVAL=24 windows
    # fine. Arms are also run SEQUENTIALLY (see main) so peak RAM == one model, not two.
    del ck
    gc.collect()
    return model, ck_res


# ============================================================================
# Eval windowing -- verbatim from finetune_raptor.py StudyWindows (train=False)
# ============================================================================
def _eval_centers(mask, D, k):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        raise ValueError("study has fewer than 3 valid slices (training contract)")
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]


def eval_windows(vol, mask, k, res, norm=NORM):
    validate_volume(vol, mask)
    D = vol.shape[0]
    cs = _eval_centers(mask, D, k)
    wins = np.empty((len(cs), 3, res, res), np.float32)
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear",
                              align_corners=False)[0]
        wins[j] = t.numpy()
    x = torch.from_numpy(wins)
    if norm == "imagenet":
        x = (x - _MEAN) / _STD
    return x


@torch.no_grad()
def infer_probs(model, xwins, device, telemetry=None):
    x = xwins.unsqueeze(0).to(device)
    use_cuda = device != "cpu" and str(device).startswith("cuda")
    if use_cuda:
        # fp16 conv on T4 is fully cuDNN-supported (bf16 is NOT -> "no engine").
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                o = _model_probabilities(model, x)
        except RuntimeError as exc:
            record_event(telemetry, "fp16_to_fp32", error=type(exc).__name__,
                         message=str(exc))
            # A retry is observable; its result still must pass the raw gate.
            torch.cuda.empty_cache()
            o = _model_probabilities(model, x)
        result = o[0].cpu().numpy()
        validate_probabilities(result, (len(LAB),), "GPU prediction")
        return result
    o = _model_probabilities(model, x)
    result = o[0].cpu().numpy()
    validate_probabilities(result, (len(LAB),), "fp32 prediction")
    return result


def _model_probabilities(model, x):
    logits = model(x).float()
    if tuple(logits.shape) != (1, len(LAB)):
        raise ValueError(f"model logits have invalid shape: {tuple(logits.shape)}")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("model logits contain non-finite values")
    return torch.sigmoid(logits)


def record_event(telemetry, event, **details):
    """Count every event, retain a bounded sample, and include the active arm/study."""
    if telemetry is None:
        return
    counts = telemetry.setdefault("counts", {})
    counts[event] = counts.get(event, 0) + 1
    samples = telemetry.setdefault("events", [])
    if len(samples) < 100:
        samples.append({"event": event, **telemetry.get("context", {}), **details})


def validate_volume(vol, mask):
    vol, mask = np.asarray(vol), np.asarray(mask)
    if vol.shape != (MAXS, IMG, IMG) or vol.dtype != np.uint8:
        raise ValueError(f"invalid volume contract: shape={vol.shape}, dtype={vol.dtype}")
    if mask.shape != (MAXS,) or not np.isin(mask, [0, 1]).all():
        raise ValueError("invalid mask: expected one binary entry per slice")
    occupied = np.any(vol != 0, axis=(1, 2))
    if not np.array_equal(mask > 0, occupied):
        raise ValueError("mask does not match nonzero volume slices")
    if int(np.count_nonzero(mask)) < 3:
        raise ValueError("study has fewer than 3 valid slices (training contract)")


def validate_probabilities(values, expected_shape, context="raw probabilities"):
    values = np.asarray(values)
    if values.shape != expected_shape:
        raise ValueError(f"{context}: expected shape {expected_shape}, got {values.shape}")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError(f"{context}: real numeric dtype required")
    if not np.isfinite(values).all():
        raise ValueError(f"{context}: non-finite values")
    if ((values < 0) | (values > 1)).any():
        raise ValueError(f"{context}: probabilities outside [0, 1]")
    return values


def fuse_probabilities(arm_probs, arms, n_studies):
    """Validate every raw arm before ranking; preserve kernel-35 ties and weights."""
    validate_arms(arms)
    if n_studies < 1 or len(arm_probs) != len(arms):
        raise ValueError("empty studies or raw arm count mismatch")
    raw = [validate_probabilities(p, (n_studies, len(LAB)), f"arm {index}")
           for index, p in enumerate(arm_probs)]
    model_ranks = [rankpct(p) for p in raw]
    own_indices = [i for i, arm in enumerate(arms) if arm["kind"] == "own"]
    public_index = next(i for i, arm in enumerate(arms) if arm["kind"] == "public")
    own_component = np.mean(np.stack([model_ranks[i] for i in own_indices]), axis=0)
    public_component = model_ranks[public_index]
    ranks = 0.5 * own_component + 0.5 * public_component
    validate_probabilities(ranks, (n_studies, len(LAB)), "rank blend")
    return model_ranks, own_component, public_component, ranks


def validate_arms(arms):
    own = [arm for arm in arms if arm["kind"] == "own"]
    public = [arm for arm in arms if arm["kind"] == "public"]
    if len(arms) != 4 or len(own) != 3 or {arm["fold"] for arm in own} != {0, 1, 2}:
        raise ValueError("exactly three own folds 0/1/2 are required")
    if len(public) != 1:
        raise ValueError("exactly one public component is required")


def rankpct(x):
    """Per-column average-tie percentile ranks in [0,1] over the full test set."""
    x = np.asarray(x)
    if x.ndim != 2 or not all(x.shape):
        raise ValueError("rank input must be a nonempty two-dimensional matrix")
    if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise ValueError("rank input must have a real numeric dtype")
    if not np.isfinite(x).all():
        raise ValueError("rank input contains non-finite values")
    x = x.astype(np.float64)
    out = np.empty_like(x)
    for column in range(x.shape[1]):
        order = np.argsort(x[:, column], kind="mergesort")
        sorted_values = x[order, column]
        ranked = np.empty(len(order), dtype=np.float64)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and sorted_values[end] == sorted_values[start]:
                end += 1
            ranked[order[start:end]] = (start + end - 1) / 2.0
            start = end
        out[:, column] = ranked / max(1, len(order) - 1)
    return out


# ============================================================================
# Preprocessing -- verbatim from kprep2/dino_preprocess.py, retargeted to TEST
# ============================================================================
def _make_reader(telemetry=None):
    import pydicom, cv2
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def order_and_meta(sdir):
        fs = glob.glob(sdir + "/*.dcm"); recs = []; ps_list = []
        for f in fs:
            try:
                h = pydicom.dcmread(f, stop_before_pixels=True)
                iop = getattr(h, 'ImageOrientationPatient', None)
                ipp = getattr(h, 'ImagePositionPatient', None)
                if iop is not None and ipp is not None and len(iop) == 6:
                    r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                    n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))
                else:
                    pos = float(getattr(h, 'InstanceNumber', 0) or 0)
                ps = getattr(h, 'PixelSpacing', None); ps = float(ps[0]) if ps is not None else 0.5
                ps_list.append(ps); recs.append((pos, f, ps))
            except Exception as exc:
                record_event(telemetry, "dicom_header_error", file=os.path.basename(f),
                             error=type(exc).__name__, message=str(exc))
                recs.append((0.0, f, 0.5))
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return [(f, ps) for _, f, ps in recs], med_ps

    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps):
        h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
        cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)

    return order_and_meta, read_px, mm_crop_resize


def _pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study(sid, ser_records, tsdir, reader, telemetry=None):
    order_and_meta, read_px, mm_crop_resize = reader
    rows = ser_records.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0; used = set()
    for plane, fluid, k in SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            record_event(telemetry, "missing_series_slot", plane=plane, fluid=fluid)
            idx += k; continue
        used.add(r['SeriesInstanceUID'])
        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            record_event(telemetry, "empty_series", series=r['SeriesInstanceUID'])
            idx += k; continue
        # wide span: the collateral ligaments and lateral meniscus live in the
        # peripheral slices the old 0.15-0.85 crop threw away. Must match the corpus
        # the weights were trained on (knee_corpus_v2.py, SPAN_LO/SPAN_HI).
        n = len(files); lo, hi = int(n * 0.06), int(n * 0.94) - 1; hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs = []; pss = []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                pixels = read_px(fp)
                if pixels.ndim != 2 or not pixels.size or not np.isfinite(pixels).all():
                    raise ValueError("DICOM pixels must be a nonempty finite 2D array")
                arrs.append(pixels); pss.append(ps)
            except Exception as exc:
                record_event(telemetry, "dicom_pixel_error", file=os.path.basename(fp),
                             error=type(exc).__name__, message=str(exc))
                arrs.append(None); pss.append(med_ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if idx >= MAXS: break
            if a is None: idx += 1; continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
            vol[idx] = (aw * 255).astype(np.uint8); idx += 1
        if idx >= MAXS: break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


# ============================================================================
# Test-root discovery + weights + main
# ============================================================================
def find_test_root():
    cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
             "/kaggle/input/rsna-knee-abnormality-detection"]
    for b in cands:
        if os.path.exists(b + "/test.csv"):
            return b
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f and (os.path.isdir(d + "/test_series") or os.path.isdir(d + "/test_images")):
            return d
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f:
            return d
    raise RuntimeError("no test root under /kaggle/input")


def find_weight_file(fname):
    # Private kernel outputs may be nested below a version/output directory. Walk all
    # non-competition mounts, but prune the competition tree before descending so the
    # DICOM corpus is never scanned. Reject ambiguous copies.
    hits = []
    for root, directories, files in os.walk("/kaggle/input"):
        parts = {part.casefold() for part in root.replace("\\", "/").split("/")}
        if "competitions" in parts or "rsna-knee-abnormality-detection" in parts:
            directories[:] = []
            continue
        if fname in files:
            hits.append(os.path.join(root, fname))
    hits = sorted(set(hits))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {fname}, found {hits}")
    return hits[0]


def verify_weight_file(path, expected):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    actual = h.hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"checkpoint hash mismatch for {os.path.basename(path)}: "
            f"expected {expected}, got {actual}"
        )
    print(f"[integrity] {os.path.basename(path)} sha256={actual}", flush=True)


def predict_studies(test_ids, ser_records, tsdir, reader, device, telemetry, arms=None):
    """Produce all raw probabilities or raise, with no replacement predictions."""
    arms = ARMS if arms is None else arms
    validate_arms(arms)
    arm_probs = [np.full((len(test_ids), len(LAB)), np.nan, np.float32) for _ in arms]
    for a, arm in enumerate(arms):
        telemetry["context"] = {"arm": a}
        wp = find_weight_file(arm["file"])
        verify_weight_file(wp, EXPECTED_WEIGHT_SHA256[arm["file"]])
        model, res = load_model(wp, arm["arch"], arm["res"], device)
        try:
            for i, sid in enumerate(test_ids):
                telemetry["context"] = {"arm": a, "study_index": i, "study_id": sid}
                try:
                    vol, mask = build_study(sid, ser_records, tsdir, reader, telemetry)
                    validate_volume(vol, mask)
                    xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM)
                    prediction = infer_probs(model, xw, device, telemetry)
                    validate_probabilities(prediction, (len(LAB),), f"arm {a} study {i}")
                    arm_probs[a][i] = prediction
                    del vol, mask, xw
                except Exception as exc:
                    record_event(telemetry, "study_failure", error=type(exc).__name__,
                                 message=str(exc))
                    raise RuntimeError(f"arm {a} study {i} {sid}: inference rejected") from exc
                if (i + 1) % 100 == 0 or i + 1 == len(test_ids):
                    print(f"[arm {a}] {i + 1}/{len(test_ids)}", flush=True)
        finally:
            del model
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    telemetry.pop("context", None)
    return arm_probs


def write_result(test_ids, sub_cols, arm_probs, output_dir, telemetry, summary=None):
    """Validate raw arrays and identities, then publish the CSV as the final step."""
    import pandas as pd
    if telemetry.get("counts", {}).get("study_failure", 0):
        raise ValueError("study failures are not allowed in final inference")
    if not test_ids or len(test_ids) != len(set(test_ids)):
        raise ValueError("empty or duplicate test UIDs")
    if sub_cols != ["StudyInstanceUID"] + LAB:
        raise ValueError("sample label/order contract differs from fixed model labels")
    model_ranks, own_component, public_component, ranks = fuse_probabilities(
        arm_probs, ARMS, len(test_ids))
    sub = pd.DataFrame(ranks.astype(np.float32), columns=LAB)
    sub.insert(0, "StudyInstanceUID", test_ids)
    np.savez_compressed(
        os.path.join(output_dir, "guarded_predictions.npz"),
        raw=np.stack(arm_probs).astype(np.float32),
        ranked_models=np.stack(model_ranks).astype(np.float32),
        own_component=own_component.astype(np.float32),
        public_component=public_component.astype(np.float32),
        blend=ranks.astype(np.float32), ids=np.asarray(test_ids),
    )
    # A partial serialization never receives the submission filename.
    pending = os.path.join(output_dir, "submission.csv.pending")
    out = os.path.join(output_dir, "submission.csv")
    sub.to_csv(pending, index=False)
    if summary is not None:
        with open(os.path.join(output_dir, "guarded_summary.json"), "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
    os.replace(pending, out)
    return out


def main(root=None, output_dir="/kaggle/working"):
    import pandas as pd
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu = torch.cuda.device_count()
    print(f"device {dev} | gpus {ngpu} | torch {torch.__version__}", flush=True)
    assert dev == "cuda" and ngpu >= 1, "anchor requires a Kaggle GPU runtime"

    ROOT = root or find_test_root()
    tsdir = ROOT + "/test_series"
    if not os.path.isdir(tsdir):
        tsdir = ROOT + "/test_images"
    print("test root:", ROOT, "| series dir:", tsdir, flush=True)

    # Deliberately load only the identifier. Free text is not an inference input.
    test = pd.read_csv(ROOT + "/test.csv", usecols=["StudyInstanceUID"])
    test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)
    test_ids = test["StudyInstanceUID"].tolist()
    assert test_ids and len(test_ids) == len(set(test_ids)), "empty or duplicate test UIDs"
    if DRY_RUN:
        raise ValueError("kernel 36 supports only the complete three-fold contract")
    own_arms = [arm for arm in ARMS if arm["kind"] == "own"]
    validate_arms(ARMS)
    tser = pd.read_csv(ROOT + "/test_series.csv")
    tser["StudyInstanceUID"] = tser["StudyInstanceUID"].astype(str)
    tser["SeriesInstanceUID"] = tser["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID")}
    print(f"test studies {len(test_ids)} | test series {len(tser)}", flush=True)

    sub_cols = ["StudyInstanceUID"] + LAB
    ssub = os.path.join(ROOT, "sample_submission.csv")
    assert os.path.exists(ssub), "sample_submission.csv is required"
    sample = pd.read_csv(ssub)
    sub_cols = list(sample.columns)
    assert len(sample) == len(test_ids), "test/sample row-count mismatch"
    assert sample["StudyInstanceUID"].astype(str).tolist() == test_ids, \
        "test/sample UID order mismatch"

    N = len(test_ids)
    os.makedirs(output_dir, exist_ok=True)
    # Refuse a stale CSV rather than allow a failed rerun to masquerade as success.
    if os.path.exists(os.path.join(output_dir, "submission.csv")):
        raise FileExistsError("output directory already contains submission.csv; use a fresh directory")
    telemetry = {"counts": {}, "events": []}
    summary = {
        "status": "GUARDED_INFERENCE_COMPLETE",
        "test_studies": N,
        "reports_used": False,
        "study_failures": [],
        "telemetry": telemetry,
        "raw_validation": "shape, real numeric, finite, [0,1] before any ranking",
        "volume_validation": "uint8 64x336x336, binary matching mask, >=3 nonzero slices",
        "ranking_scope": "complete test set per model, never batch or chunk",
        "rank_ties": "average",
        "own_folds": [arm["fold"] for arm in own_arms],
        "own_fold_weights_within_component": [1 / len(own_arms)] * len(own_arms),
        "component_weights": {"own": 0.5, "public": 0.5},
        "arms": [{k: v for k, v in arm.items() if k != "res"} for arm in ARMS],
        "expected_weight_sha256": EXPECTED_WEIGHT_SHA256,
        "output": "submission.csv",
        "elapsed_seconds": time.time() - t0,
        "torch": torch.__version__,
        "timm": timm.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    try:
        reader = _make_reader(telemetry)
        arm_probs = predict_studies(test_ids, SER, tsdir, reader, dev, telemetry)
        summary["elapsed_seconds"] = time.time() - t0
        out = write_result(test_ids, sub_cols, arm_probs, output_dir, telemetry, summary)
    except Exception as exc:
        rejection = {"status": "INFERENCE_REJECTED_NO_SUBMISSION", "telemetry": telemetry,
                     "error": type(exc).__name__, "message": str(exc),
                     "elapsed_seconds": time.time() - t0}
        with open(os.path.join(output_dir, "guarded_summary.json"), "w", encoding="utf-8") as stream:
            json.dump(rejection, stream, indent=2)
        raise
    print("wrote", out, "|", N, "rows x", len(sub_cols), "cols", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="Competition root (default: discover Kaggle mount)")
    parser.add_argument("--output-dir", default="/kaggle/working")
    args = parser.parse_args()
    main(root=args.root, output_dir=args.output_dir)
