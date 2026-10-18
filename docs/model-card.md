# Modelo y límites

## Uso previsto
Investigación de clasificación de doce hallazgos en RM de rodilla. No es un producto sanitario ni una herramienta diagnóstica validada.

## Receta final
CoAtNet con cinco folds, checkpoints de época 8 y backbone coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k. La plantilla exportada declara IMG=336 y42 ventanas; el384 del nombre identifica el backbone, no la resolución efectiva. Agregación por rangos por etiqueta en la población completa, pesos iguales por fold, sin TTA. La plantilla conserva CONFIG={} y necesita configuración/pesos/datos autorizados; no funciona directamente como inferencia completa.

## Evaluación
Comparación pareada común:4349 estudios weak O2 y 58 gold, bootstrap 2000. CoAtNet:0.780478 weak y0.836965 gold; N1:0.704753 y0.720249. No son puntuaciones del leaderboard. Gold había sido usado en desarrollo; muestra pequeña, comparación no causal de arquitecturas y sin validación clínica externa. F0/F1 conservan su validación propia.

## Entrega
CoAtNet propio aceptado en Kaggle y seleccionado explícitamente como único envío el18-oct2026, con score público0.869. No demuestra posición final, premio ni tiempo del conjunto oculto. La tabla pública se calculaba con30% del test y la privada con70% restante. N1 quedó como incidencia HTTP sin aceptación confirmada, sin repetir la solicitud. No usar el mejor score histórico0.924 de un ancla pública como resultado propio de este modelo.

## Privacidad
No se publican estudios, informes, UID reales, particiones, etiquetas individuales, predicciones, pesos, credenciales ni capturas autenticadas. Los nombres de campos DICOM presentes en el código son un esquema, no registros clínicos.
