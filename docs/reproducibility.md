# Qué se puede reproducir

## Demostración pública
Python 3.11 y NumPy 2.4.4. Desde la raíz:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-demo.txt
python -B -m unittest tests.test_coatnet_tie_metrics tests.test_coatnet_tie_auc_bound
```

Las pruebas usan matrices sintéticas, sin imágenes, pesos, etiquetas de pacientes, GPU, red ni envíos. Extraen solo el AST de rankpct de las fuentes históricas22/36; no importan ni ejecutan esos scripts de inferencia. Los módulos prueban empates, invariancia, orden y cotas de cambio AUC sintéticas; no prueban rendimiento clínico.

## Reproducción de la inferencia completa
La campaña privada fijó fuentes, checkpoints, dependencias y entorno por hash. Reproducciones independientes desde ZIP generaron CSV idénticos en tres estudios visibles; benchmarks separados cubrieron1500 estudios y8303 series. El benchmark CoAtNet usó una conciliación temporal explícita conservando el fallo original. Nada de ello acredita reproducción completa del entrenamiento o tiempo del test oculto.

Esta edición no incluye los pesos/datos/CONFIG necesarios. Los scripts son referencias técnicas, no un comando turnkey. No instalar requirements-demo como entorno de inferencia: faltan deliberadamente PyTorch, timm, pandas y dependencias DICOM. No se ejecutan jobs remotos ni se proporcionan credenciales.

## Procedencia de fuentes
[provenance.json](provenance.json) registra hashes SHA256 de las siete fuentes exportadas. El repositorio usa historial Git nuevo; no contiene el historial operacional privado. No se distribuyen los recibos privados: los resultados de campaña son un resumen documental y no pueden reauditarse íntegramente con esta edición pública.
