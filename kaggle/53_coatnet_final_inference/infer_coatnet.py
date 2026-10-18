"""Five fixed epoch8 CoAtNet folds: offline DICOM to global ranks to CSV.

Scientific functions below are copied exactly from frozen kernel36. The public
component and old selected epochs are excluded. DICOM preprocessing originates
in Dread Development's Apache-2.0 V1 scriptVersionId344173275 (existing attribution).
"""
import time
PROCESS_STARTED = time.perf_counter()
import os, sys, glob, json, gc, hashlib, shutil, platform
from datetime import datetime, timezone
from pathlib import Path
STARTED_AT_UTC = datetime.now(timezone.utc).isoformat()
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import timm
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
CONFIG = {}
IMG = 336
CROP_MM = 140.0
SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
         ("Coronal", 0, 8), ("Axial", -1, 12)]
MAXS = sum(s[2] for s in SLOTS)
K_EVAL = 42
NORM = "imagenet"
ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
SERIES_COLUMNS = ["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"]

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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def output_record(path):
    path = Path(path)
    return dict(path=path.name, sha256=sha(path), bytes=path.stat().st_size)


def validate_ids(ids):
    require(bool(ids) and all(isinstance(x, str) and x and x not in {".", ".."}
            and not any(c in x for c in "/\\\x00") for x in ids)
            and len(ids) == len(set(ids)), "invalid or duplicate study identifiers")


def validate_series(frame, ids):
    require(list(frame.columns) == SERIES_COLUMNS and not frame.isna().any().any(), "series metadata projection")
    require(set(frame.StudyInstanceUID) == set(ids) and not frame.duplicated(["StudyInstanceUID", "SeriesInstanceUID"]).any(),
            "series coverage or identity mismatch")
    validate_ids(frame.SeriesInstanceUID.drop_duplicates().tolist())
    require(frame.Anatomical_Plane.isin(["Sagittal", "Coronal", "Axial"]).all()
            and frame.Fluid_Sensitive.isin([0, 1]).all(), "invalid technical series metadata")
    return {sid: rows.to_dict("records") for sid, rows in frame.groupby("StudyInstanceUID", sort=False)}


def load_population(root, config):
    root = Path(root)
    if config["mode"] == "benchmark":
        ids = config["population"]["study_ids"]
        frame = pd.DataFrame(config["population"]["series_records"], columns=SERIES_COLUMNS)
        sub_cols = ["StudyInstanceUID"] + LAB
        input_hashes = dict(config["population"]["input_hashes"])
        tsdir = root / "train_series"
        if not tsdir.is_dir():
            tsdir = root / "train_images"
    else:
        # Never read Report or train targets. The sample values are ignored.
        test = pd.read_csv(root / "test.csv", usecols=["StudyInstanceUID"], dtype=str)
        ids = test.StudyInstanceUID.tolist()
        sample = pd.read_csv(root / "sample_submission.csv", dtype={"StudyInstanceUID": str})
        sub_cols = list(sample.columns)
        require(sample.StudyInstanceUID.tolist() == ids, "test/sample UID order mismatch")
        frame = pd.read_csv(root / "test_series.csv", usecols=SERIES_COLUMNS,
                            dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})[SERIES_COLUMNS]
        input_hashes = {name: sha(root / name) for name in ("test.csv", "test_series.csv", "sample_submission.csv")}
        tsdir = root / "test_series"
        if not tsdir.is_dir():
            tsdir = root / "test_images"
    validate_ids(ids)
    require(sub_cols == ["StudyInstanceUID"] + LAB, "exact submission columns required")
    require(tsdir.is_dir(), "DICOM directory absent")
    return ids, validate_series(frame, ids), tsdir, input_hashes


def validate_config(config):
    require(config.get("schema") == "coatnet-final-inference-config-v1"
            and config.get("mode") in {"benchmark", "competition"}, "unprepared source")
    require([e["fold"] for e in config["endpoints"]] == list(range(5)), "five unique ordered folds required")
    require(all(e["epoch"] == 8 and e["filename"] == f"fold{e['fold']}_epoch8_model.pt"
            and len(e["sha256"]) == 64 for e in config["endpoints"]), "fixed epoch8 checkpoints required")
    require(len({e["sha256"] for e in config["endpoints"]}) == 5, "checkpoints must be distinct")
    require(config["recipe"] == dict(epoch=8, folds=list(range(5)), windows=42, resolution=384,
            normalization="imagenet", rank_scope="entire_population_per_model", ties="average",
            fold_weights=[0.2] * 5, tta=False, reports_used=False, training=False), "recipe drift")
    require(datetime.fromisoformat(STARTED_AT_UTC) >= datetime.fromisoformat(config["frozen_at_utc"]),
            "inference predates five-fold freeze")
    if config["mode"] == "benchmark":
        p = config["population"]
        validate_ids(p["study_ids"])
        require(len(p["study_ids"]) == 1500 and p["gold_excluded"] is True, "full fixed weak cohort required")


def validate_checkpoint(checkpoint, fold):
    require(checkpoint["epoch"] == checkpoint["endpoint_epoch"] == 8
            and checkpoint["outer_fold"] == fold and checkpoint["arm"] == "single_cycle"
            and checkpoint["architecture"] == ARCH and checkpoint["seed"] == 20260829
            and checkpoint["epoch_selection_performed"] is False and "model_state_dict" in checkpoint,
            "checkpoint recipe differs from audited epoch8")
    require(all(torch.isfinite(v).all().item() for v in checkpoint["model_state_dict"].values()),
            "non-finite checkpoint tensors")


def predict_cached(ids, volumes, masks, endpoints, device, telemetry, phase_seconds):
    raw = np.full((5, len(ids), len(LAB)), np.nan, dtype=np.float32)
    for endpoint in endpoints:
        fold, started = endpoint["fold"], time.perf_counter()
        checkpoint_path = find_weight_file(endpoint["filename"])
        verify_weight_file(checkpoint_path, endpoint["sha256"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        validate_checkpoint(checkpoint, fold)
        del checkpoint
        model, resolution = load_model(checkpoint_path, ARCH, 384, device)
        require(resolution == 384, "checkpoint resolution changed")
        try:
            for i, sid in enumerate(ids):
                telemetry["context"] = dict(fold=fold, study_index=i, study_id=sid)
                windows = eval_windows(volumes[i], masks[i], K_EVAL, resolution, NORM)
                prediction = infer_probs(model, windows, device, telemetry)
                validate_probabilities(prediction, (len(LAB),), f"fold{fold} study{i}")
                raw[fold, i] = prediction
                del windows
                if (i + 1) % 100 == 0 or i + 1 == len(ids):
                    print(f"[fold {fold}] {i + 1}/{len(ids)}", flush=True)
        finally:
            del model
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        phase_seconds[f"fold{fold}_load_windows_forward"] = time.perf_counter() - started
    telemetry.pop("context", None)
    return raw


def write_submission(ids, raw, output):
    validate_ids(ids)
    validate_probabilities(raw, (5, len(ids), len(LAB)), "fivefold raw scores")
    ranked = np.stack([rankpct(scores) for scores in raw])
    mean = ranked.mean(axis=0)
    validate_probabilities(mean, (len(ids), len(LAB)), "fivefold mean ranks")
    output = Path(output)
    require(not (output / "submission.csv").exists() and not (output / "coatnet_predictions.npz").exists(),
            "never overwrite prior predictions")
    np.savez_compressed(output / "coatnet_predictions.npz", ids=np.asarray(ids), raw=raw,
                        ranked=ranked, prediction=mean.astype(np.float32))
    frame = pd.DataFrame(mean.astype(np.float32), columns=LAB)
    frame.insert(0, "StudyInstanceUID", ids)
    pending = output / "submission.csv.pending"
    with pending.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)
    pending.replace(output / "submission.csv")
    return {name: output_record(output / name) for name in ("submission.csv", "coatnet_predictions.npz")}


def prepare_volumes(ids, series, tsdir, output, telemetry):
    # uint8 caches preserve the exact result of the frozen preprocessor and avoid
    # decoding every DICOM five times. One model and one study's windows live on GPU.
    cache = output / "temporary_volume_cache"
    cache.mkdir(exist_ok=False)
    needed = len(ids) * MAXS * IMG * IMG + len(ids) * MAXS
    require(shutil.disk_usage(output).free >= needed + 2 * 1024**3, "insufficient cold-cache disk headroom")
    volumes = np.lib.format.open_memmap(cache / "volumes.npy", mode="w+", dtype=np.uint8,
                                       shape=(len(ids), MAXS, IMG, IMG))
    masks = np.lib.format.open_memmap(cache / "masks.npy", mode="w+", dtype=np.uint8, shape=(len(ids), MAXS))
    order_and_meta, read_px, resize = _make_reader(telemetry)
    inventory, payloads, header_results = [], {}, []
    # Inventory every announced series. Pixel content hashes cover the selected
    # DICOMs; header-derived ordering/spacing for every read series is also recorded.
    for sid in ids:
        for row in series[sid]:
            directory = tsdir / sid / row["SeriesInstanceUID"]
            files = sorted(directory.glob("*.dcm"))
            inventory.append(dict(study=sid, series=row["SeriesInstanceUID"],
                                  files=[dict(name=p.name, bytes=p.stat().st_size) for p in files]))
    def traced_order(directory):
        records, spacing = order_and_meta(directory)
        header_results.append(dict(directory=Path(directory).relative_to(tsdir).as_posix(), median_spacing=spacing,
                                   ordered=[dict(file=Path(f).name, spacing=p) for f, p in records]))
        return records, spacing
    def traced_pixels(filename):
        path = Path(filename)
        key = path.relative_to(tsdir).as_posix()
        if key not in payloads:
            payloads[key] = dict(sha256=sha(path), bytes=path.stat().st_size)
        return read_px(filename)
    for i, sid in enumerate(ids):
        telemetry["context"] = dict(study_index=i, study_id=sid)
        volume, mask = build_study(sid, series, str(tsdir), (traced_order, traced_pixels, resize), telemetry)
        validate_volume(volume, mask)
        volumes[i], masks[i] = volume, mask
    volumes.flush()
    masks.flush()
    save(output / "dicom_input_inventory.json", dict(series=inventory, header_results=header_results,
                                                     selected_pixel_files=payloads))
    return volumes, masks, dict(cache_bytes=needed, selected_pixel_files=len(payloads),
                                series=len(inventory), dicom_files=sum(len(r["files"]) for r in inventory))


def find_root(mode):
    for root in (Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
                 Path("/kaggle/input/rsna-knee-abnormality-detection")):
        if (root / ("train_series.csv" if mode == "benchmark" else "test.csv")).exists():
            return root
    raise FileNotFoundError("competition DICOM mount absent")


def run(root, output):
    validate_config(CONFIG)
    require(torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0), "audited T4 runtime required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    require(not any((output / name).exists() for name in ("submission.csv", "coatnet_runtime.json", "coatnet_predictions.npz")),
            "fresh output directory required")
    phases, telemetry = {}, {"counts": {}, "events": []}
    volumes = masks = None
    try:
        started = time.perf_counter()
        ids, series, tsdir, input_hashes = load_population(root, CONFIG)
        phases["metadata"] = time.perf_counter() - started
        started = time.perf_counter()
        volumes, masks, resources = prepare_volumes(ids, series, tsdir, output, telemetry)
        phases["dicom_read_preprocess_cache_inventory"] = time.perf_counter() - started
        raw = predict_cached(ids, volumes, masks, CONFIG["endpoints"], "cuda", telemetry, phases)
        started = time.perf_counter()
        outputs = write_submission(ids, raw, output)
        outputs["dicom_input_inventory.json"] = output_record(output / "dicom_input_inventory.json")
        phases["global_rank_csv_and_hashes"] = time.perf_counter() - started
        import cv2, pydicom, resource
        resources.update(peak_ram_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                         peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                         peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                         cuda_total_memory_bytes=int(torch.cuda.get_device_properties(0).total_memory),
                         gpu_count=int(torch.cuda.device_count()), models_simultaneously_on_gpu=1)
        started = time.perf_counter()
        del volumes, masks
        volumes = masks = None
        gc.collect()
        cache = output / "temporary_volume_cache"
        for name in ("volumes.npy", "masks.npy"):
            (cache / name).unlink()
        cache.rmdir()
        phases["cache_cleanup"] = time.perf_counter() - started
        source_sha256 = sha(__file__)
        elapsed = time.perf_counter() - PROCESS_STARTED
        summary = dict(schema="integral-inference-runtime-v1", model="COATNET", status="COMPLETE",
            mode=CONFIG["mode"], config=CONFIG, source_sha256=source_sha256, freeze_sha256=CONFIG["freeze_sha256"],
            population_scope="preregistered_1500_weak_metadata" if CONFIG["mode"] == "benchmark" else "competition_test",
            population_sha256=CONFIG["population"]["manifest_sha256"] if CONFIG["mode"] == "benchmark" else input_hashes["test.csv"],
            study_count=len(ids), checkpoint_sha256=[e["sha256"] for e in CONFIG["endpoints"]], input_hashes=input_hashes,
            ordered_uid_sha256=hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
            elapsed_seconds=elapsed, phase_seconds=phases, inference_started_at_utc=STARTED_AT_UTC,
            environment=dict(python=sys.version, platform=platform.platform(), torch=str(torch.__version__),
                timm=timm.__version__, numpy=np.__version__, pandas=pd.__version__, pydicom=pydicom.__version__,
                opencv=cv2.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0)),
            resources=resources, outputs=outputs, telemetry=telemetry,
            runtime_within_22680_seconds=elapsed <= 22680, full_hidden_test_runtime_verified=False,
            timer_scope="first Python statement through output serialization and hashes; excludes interpreter/container startup",
            reports_used=False, targets_read=False, training=False, tta=False,
            rank_scope="entire_population_per_model", rank_ties="average", fold_weights=[0.2] * 5)
        save(output / "coatnet_runtime.json", summary)
        return summary
    except Exception as exc:
        # A failed run must not retain a success-shaped CSV.
        pending_csv = output / "submission.csv"
        if pending_csv.exists():
            pending_csv.replace(output / "rejected_submission.csv")
        save(output / "coatnet_rejection.json", dict(status="REJECTED", error=type(exc).__name__, message=str(exc),
             elapsed_seconds=time.perf_counter()-PROCESS_STARTED, telemetry=telemetry))
        raise
    finally:
        del volumes, masks
        gc.collect()
        # Delete only this function's fixed cache files under its output directory.
        cache = output / "temporary_volume_cache"
        if cache.is_dir():
            for name in ("volumes.npy", "masks.npy"):
                (cache / name).unlink(missing_ok=True)
            cache.rmdir()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    print(json.dumps(run(args.root or find_root(CONFIG.get("mode")), args.output), indent=2), flush=True)
