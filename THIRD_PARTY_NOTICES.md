# Atribuciones y alcance de licencia

El código de inferencia adapta trabajo de [Dread Development](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model) (Apache-2.0), identificado como tal en las fuentes y en NOTICE. Los scripts22/36 son referencias históricas usadas por las pruebas de ranking; no son la entrega final ni se reivindican como modelos enteramente originales.

La arquitectura se instancia mediante [timm](https://github.com/huggingface/pytorch-image-models) (Apache-2.0). PyTorch usa licencia BSD-style; NumPy BSD-3-Clause; pandas BSD-3-Clause; pydicom MIT; OpenCV4.x Apache-2.0. Son dependencias externas, no vendorizadas aquí. Consulte las licencias de las versiones que instale.

La licencia Apache-2.0 del repositorio cubre código/documentación distribuida según sus atribuciones; no concede derechos sobre DICOM, informes, datos de competición, checkpoints o modelos externos. No se distribuyen datos ni pesos. El acceso a RSNA exige cumplir las reglas de Kaggle por separado. Se omiten la rama NLI y sus pesos; no se afirma resuelta la licencia de todos los artefactos de la investigación privada.
