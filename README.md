# RSNA Knee Abnormality Detection

### Aprender a mirar, construir y aceptar los límites

Detrás de una resonancia de rodilla hay una persona. Aquí trabajamos con imágenes, código y números, pero el contexto de salud merece estar presente desde el principio: reconocer un patrón en una competición es solo una pequeña parte del camino hacia algo que pueda ayudar a alguien.

RSNA Knee Abnormality Detection reúne el trabajo realizado para explorar la detección de doce hallazgos en resonancias de rodilla, dentro de una competición de Kaggle. Este repositorio cuenta ese recorrido: lo que se consiguió construir, lo que se torció y las lecciones que quedan después de cerrar una entrega.

## Pequeños hitos en Kaggle

Aprender también deja pequeños recuerdos del camino. Estas son las **13 insignias** de participación y aprendizaje del [perfil Iber1to](https://www.kaggle.com/iber1to), verificadas el **18 de octubre de 2026**. Cada una enlaza con su certificado en Kaggle.

| | | | |
|:---:|:---:|:---:|:---:|
| [<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1488634%2Fb05e17f63f2eead904a1f04d05bb51a6%2FBadge.svg?generation=1727458564383731&amp;alt=media" width="64" height="64" alt="Competitor"><br>Competitor](https://www.kaggle.com/certification/badges/iber1to/5) | [<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1488634%2F33a465e5bfb87721b4d97d899c717eb4%2FBadge-3.svg?generation=1727462497490072&amp;alt=media" width="64" height="64" alt="Research Competitor"><br>Research Competitor](https://www.kaggle.com/certification/badges/iber1to/7) | [<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1488634%2F08aa345b8928ae96e21ee3586021500e%2FBadge-8.svg?generation=1727462754402917&amp;alt=media" width="64" height="64" alt="Code Submitter"><br>Code Submitter](https://www.kaggle.com/certification/badges/iber1to/13) | [<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1488634%2F09e1f99bdf3222934ad7769409ec3f6d%2FBadge-26.svg?generation=1727468059623106&amp;alt=media" width="64" height="64" alt="Python Coder"><br>Python Coder](https://www.kaggle.com/certification/badges/iber1to/30) |
| [<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1488634%2F4320b13590f043b8c58421eb25f04235%2FBadge-33.svg?generation=1727468143701354&amp;alt=media" width="64" height="64" alt="API Notebook Creator"><br>API Notebook Creator](https://www.kaggle.com/certification/badges/iber1to/34) | [<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1488634%2F42167cf9c1c0a74d4e1c7503c010c82d%2FBadge-31.svg?generation=1727468175250681&amp;alt=media" width="64" height="64" alt="Colab Coder"><br>Colab Coder](https://www.kaggle.com/certification/badges/iber1to/36) | [<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1488634%2F2df508502fba7fc8937f3847e46b8ec4%2FBadge-29.svg?generation=1727468233931525&amp;alt=media" width="64" height="64" alt="Notebook Modeler"><br>Notebook Modeler](https://www.kaggle.com/certification/badges/iber1to/39) | [<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1488634%2F28e0e70842ce6972f4d68f5b6ecd549a%2FBadge-12.svg?generation=1727462988946700&amp;alt=media" width="64" height="64" alt="Dataset Creator"><br>Dataset Creator](https://www.kaggle.com/certification/badges/iber1to/17) |
| [<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1488634%2F4bcf6474cdf3503864419523d22b6019%2FBadge-15.svg?generation=1727463114221819&amp;alt=media" width="64" height="64" alt="API Dataset Creator"><br>API Dataset Creator](https://www.kaggle.com/certification/badges/iber1to/20) | [<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1488634%2Feb028e2f7e6b3e2bef2b3682f6919224%2FBadge-25.svg?generation=1727468044501552&amp;alt=media" width="64" height="64" alt="Competition Modeler"><br>Competition Modeler](https://www.kaggle.com/certification/badges/iber1to/29) | [<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F10489106%2F8235ef48b7e37d36d6e081d112f303de%2FBadge-62.svg?generation=1758039755687969&amp;alt=media" width="64" height="64" alt="Kaggle Community Member"><br>Kaggle Community Member](https://www.kaggle.com/certification/badges/iber1to/101) | [<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1488634%2Fe06d05ebef3630403946b71cbdc11665%2FBadge-42.svg?generation=1727468309615075&amp;alt=media" width="64" height="64" alt="Bookmarker"><br>Bookmarker](https://www.kaggle.com/certification/badges/iber1to/43) |
| [<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1488634%2F6e139e7278fc9c33210e39baf918967f%2FBadge-53.svg?generation=1727468684057299&amp;alt=media" width="64" height="64" alt="7 Day Login Streak"><br>7 Day Login Streak](https://www.kaggle.com/certification/badges/iber1to/56) |  |  |  |

## Un camino de pruebas y decisiones

El reto parecía concreto: enseñar a un modelo a encontrar señales en imágenes médicas. Llevarlo hasta una entrega exigió mucho más que entrenarlo. Hubo que preparar las imágenes, comparar alternativas, cuidar un presupuesto de cómputo gratuito y comprobar que cada resultado correspondía a la versión que se quería evaluar.

Una parte importante del esfuerzo estuvo en ese trabajo poco visible: volver sobre una comprobación, entender una diferencia, dejar constancia de un fallo. Son tareas que rara vez aparecen en una puntuación, pero que permiten confiar un poco más en lo que se está haciendo.

El proyecto se apoyó en trabajo abierto de otras personas y añadió desarrollo propio. Las [atribuciones](THIRD_PARTY_NOTICES.md) y la [procedencia del código](docs/provenance.json) recogen esa deuda.

## El resultado, con perspectiva

El modelo propio llegó a Kaggle, fue aceptado y quedó seleccionado como entrega final. Su puntuación pública fue **0,869**, frente a un líder que rondaba **0,969** el **18 de octubre de 2026**. Quedó lejos de los primeros puestos.

Después de tanto trabajo, esa distancia también forma parte de la historia. Una entrega que funciona no garantiza el resultado que se esperaba. Aun así, el recorrido deja algo valioso: práctica para construir, criterio para evaluar y una idea más clara de lo que habría que mejorar.

También quedó una dificultad sin resolver. Una segunda alternativa, llamada N1, tuvo un error durante el envío y no se pudo confirmar su aceptación. Se conservó el registro y se decidió no repetir la solicitud para evitar duplicados. La entrega final quedó únicamente con CoAtNet.

Estos resultados describen una competición y sus comprobaciones internas. **El proyecto es experimental y no está validado para tomar decisiones clínicas.** Los detalles y límites están en la [ficha del modelo](docs/model-card.md).

## Lo que queda del aprendizaje

Comparar modelos enseñó a distinguir una mejora interna de una mejora frente a otros equipos. Revisar resultados enseñó que una muestra pequeña puede dar una confianza mayor de la que merece. Y el fallo de N1 dejó una lección concreta: guardar que algo falló no basta; hace falta conservar información útil para entender por qué.

Cerrar el proyecto también supuso aceptar hasta dónde llegaba la evidencia. Quedaron preguntas abiertas y líneas de investigación pendientes. Dejarlas escritas permite que quien llegue después entienda tanto las decisiones como sus límites.

La [retrospectiva](docs/project-retrospective.md) desarrolla estos aprendizajes.

## Explorar el trabajo

Esta edición pública comparte una selección de código, una plantilla del modelo final y **36 pruebas con datos artificiales**. Permite acercarse a algunas de las decisiones del proyecto sin acceder a información privada.

No incluye resonancias, informes, identificadores de pacientes, predicciones, pesos del modelo ni credenciales. La plantilla necesita configuración, datos y pesos autorizados por separado; las pruebas públicas no reproducen el entrenamiento ni la puntuación de Kaggle.

Para ejecutar esas pruebas con Python 3.11:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-demo.txt
python -B -m unittest tests.test_coatnet_tie_metrics tests.test_coatnet_tie_auc_bound
```

No requieren GPU ni acceso a Kaggle. El [alcance de reproducción](docs/reproducibility.md) explica qué comprueban y qué queda fuera.

<details>
<summary>Ver el recorrido técnico y las comprobaciones</summary>

![Esquema del procesamiento: las imágenes y sus metadatos pasan por la preparación, el modelo CoAtNet y la combinación de resultados para doce hallazgos.](docs/assets/pipeline.svg)

La plantilla final utiliza CoAtNet con cinco particiones de entrenamiento, checkpoints de época 8, resolución efectiva de 336 y 42 ventanas. Combina resultados mediante rangos por hallazgo, sin TTA ni entrenamiento durante la inferencia. El nombre del backbone es `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`; el 384 de ese nombre no es la resolución usada. El archivo `kaggle/53_coatnet_final_inference/infer_coatnet.py` conserva `CONFIG = {}`. Las fuentes 22/36 son referencias históricas, no entregas finales ni una fusión entregada.

La comparación interna sobre 4349 estudios weak O2 dio 0,780478 para CoAtNet y 0,704753 para N1. En los 58 estudios gold, los valores fueron 0,836965 y 0,720249. Se utilizó bootstrap pareado con 2000 remuestreos. El conjunto gold era pequeño y ya había intervenido en el desarrollo; esta comparación no demuestra superioridad clínica ni aísla el efecto de la arquitectura.

Las reproducciones desde ZIP generaron archivos idénticos en tres estudios visibles. Un benchmark separado cubrió 1500 estudios y 8303 series. Estas comprobaciones no acreditan el tiempo del conjunto oculto ni la reproducción completa del entrenamiento.

El envío propio seleccionado fue el **57319469**. El mejor resultado histórico de la cuenta, 0,924, procedía de una referencia pública externa y no corresponde al modelo propio. La clasificación pública usaba aproximadamente el 30 % del test y la privada el 70 % restante; la posición final podía cambiar. Kaggle advertía de selección automática con menos de dos plazas: solo se confirmó la selección explícita de CoAtNet en la fecha indicada.

Fuentes: [competición y envíos](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/submissions), [clasificación pública](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard) y [ficha del modelo](docs/model-card.md). Los recibos privados no se publican; las cifras de la campaña son un resumen documental y no pueden comprobarse íntegramente desde este repositorio.

</details>

**Licencia:** [Apache-2.0](LICENSE), con [atribuciones y exclusiones](THIRD_PARTY_NOTICES.md) y [NOTICE](NOTICE). El acceso a datos y pesos de la competición requiere cumplir sus condiciones por separado.
