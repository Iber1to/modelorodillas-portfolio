# ModeloRodillas

### Clasificación experimental de 12 hallazgos en resonancias de rodilla

Proyecto de portafolio sobre visión médica, inferencia DICOM y reproducibilidad. **Una entrega aceptada no equivale a un modelo clínicamente validado ni a un resultado competitivo de primer nivel.** Esta edición pública reúne código representativo y pruebas sintéticas y una valoración personal; no distribuye datos ni pesos.

![Pipeline conceptual: DICOM y metadatos pasan por preprocesado, CoAtNet de cinco folds y agregación por rangos para doce hallazgos.](docs/assets/pipeline.svg)

## Qué construimos

- Una plantilla de inferencia de cinco folds CoAtNet: lectura DICOM, selección de ventanas, preprocesado y clasificación multilabel.
- Agregación mediante rangos por etiqueta sobre la población completa, con tratamiento explícito de empates.
- Diagnósticos de invariancia y cotas matemáticas comprobados con datos sintéticos.
- Una campaña privada con versiones congeladas, hashes, control de recursos y verificación de salidas.

El trabajo combina adaptación de código abierto e ingeniería propia. La procedencia y licencias están en [NOTICE](NOTICE), [atribuciones](THIRD_PARTY_NOTICES.md) y [provenance.json](docs/provenance.json).

## Resultado y límites

Estado observado el **18 de octubre de 2026**:

| Evidencia | Resultado | Interpretación correcta |
|---|---|---|
| CoAtNet propio, envío 57319469 | Aceptado en Kaggle y seleccionado explícitamente | No garantiza clasificación privada final ni premio |
| Puntuación pública del modelo seleccionado | **0,869** | Resultado público, no métrica clínica |
| Referencia competitiva pública | Líder alrededor de 0,969 | La candidatura propia no estaba cerca de los primeros puestos |
| Comparación interna común | CoAtNet 0,780478 frente N1 0,704753, sobre 4349 weak O2 | Comparación interna; no prueba superioridad frente a rivales |
| Evaluación gold | CoAtNet 0,836965 frente N1 0,720249, sobre 58 estudios | Gold pequeño y previamente utilizado en desarrollo |
| Reproducción desde ZIP | CSV idénticos sobre 3 estudios visibles | No prueba tiempo sobre toda la población oculta |
| Benchmark operacional | 1500 estudios / 8303 series auditados | Prueba operativa, no validación clínica externa |

Fuentes: [competición y envíos](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/submissions), [clasificación pública](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard) y resumen de evidencias de la campaña privada en la [model card](docs/model-card.md). La tabla pública usaba aproximadamente 30 % del test y el resultado privado 70 %; las posiciones podían cambiar. Los recibos privados no se publican: estas cifras no son íntegramente reauditables desde este repositorio.

La comparación interna utilizó bootstrap pareado2000. No aísla causalmente la arquitectura; el gold ya había influido en el desarrollo. No afirmamos calibración clínica, validación externa ni eficacia diagnóstica. El mejor score histórico 0,924 de la cuenta procedía de un ancla pública externa y **no se presenta como resultado del modelo propio**.

N1 quedó como incidencia de envío HTTP sin aceptación confirmada. Se conservó la incertidumbre y no se repitió la solicitud para evitar duplicados. No forma parte de la entrega final. Kaggle advertía autoselección con menos de dos plazas; solo se acreditó la selección explícita de CoAtNet en esa fecha.

## Arquitectura y decisiones

La plantilla usa `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, cinco folds y endpoints de época 8. **La resolución efectiva del código exportado es336**: el384 del nombre pertenece al backbone. Se procesan 42 ventanas y se agregan resultados por rangos, sin TTA ni entrenamiento durante inferencia.

`kaggle/53_coatnet_final_inference/infer_coatnet.py` conserva `CONFIG = {}`. Es una **plantilla técnica**, no un programa de inferencia completo listo para ejecutar: faltan configuración, pesos y datos autorizados. Las fuentes22/36 se incluyen como referencias históricas de las reglas de ranking; no son la selección final ni un producto de fusión entregado.

## Ejecutar la demostración pública

Requisitos: Python 3.11 y NumPy. Desde la raíz del repositorio:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-demo.txt
python -B -m unittest tests.test_coatnet_tie_metrics tests.test_coatnet_tie_auc_bound
```

Las 36 pruebas son sintéticas: no requieren datos clínicos, pesos, GPU, acceso a Kaggle ni red durante su ejecución. Los diagnósticos extraen únicamente el AST de las funciones de ranking y no ejecutan los scripts completos. Pasar esta suite no acredita reproducción del modelo, de su entrenamiento o de su puntuación pública.

[Alcance exacto de reproducción](docs/reproducibility.md). `requirements-demo.txt` sirve para las pruebas, no es el entorno completo de inferencia.

## Qué contiene esta edición

```text
kaggle/        Plantilla final y referencias históricas de ranking
tools/        Diagnósticos de empates y cotas sintéticas
tests/        Pruebas sin datos privados
docs/         Modelo, reproducción, procedencia y retrospectiva
LICENSE       Apache-2.0, con atribuciones de terceros
```

No contiene DICOM, informes, identificadores reales de estudios, etiquetas individuales, predicciones, checkpoints, credenciales, logs de cuenta ni historial Git privado. No concede derechos sobre los datos o pesos de la competición. Para acceder a ellos hay que cumplir sus condiciones por separado.

## Aprendizajes

La principal aportación es de ingeniería: trazabilidad, reproducibilidad acotada, límites de validación explícitos y tratamiento seguro de errores. El resultado competitivo fue débil. La mejora frente a un baseline interno no justificaba afirmar que el modelo competiría por premios.

La incidencia de N1 mostró un defecto concreto de observabilidad: registrar solo el tipo de excepción pierde el diagnóstico del servidor. La solución segura fue conservar la solicitud y evitar reintentos ciegos, no ocultar el fallo. [Retrospectiva y lecciones](docs/project-retrospective.md).

## Elaboración por roles

El README se preparó con **Desarrollador**, **Diseñador gráfico** y **Evaluador**. Son roles de agentes de IA, no revisores clínicos externos. Solo el Evaluador puede aprobar la versión final; su dictamen se vincula al SHA256 exacto del README en [el registro de revisión](docs/roles.md).

**Licencia:** [Apache-2.0](LICENSE), respetando [atribuciones y exclusiones](THIRD_PARTY_NOTICES.md). Uso de investigación; no tomar decisiones clínicas con este proyecto.
