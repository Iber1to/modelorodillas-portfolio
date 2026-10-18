# Retrospectiva

## Qué aportó el proyecto
Inferencia DICOM, integración de modelos, agregación por rangos, pruebas de invariantes, control de presupuesto gratuito y trazabilidad de fuentes/artefactos. El valor de ingeniería no equivale a competitividad en Kaggle.

## Resultado honesto
El modelo propio seleccionado obtuvo0.869 público; los líderes rondaban0.969 al18-oct2026. No se obtuvo evidencia de una candidatura fuerte a premios/medallas. La mejora interna frente a N1 no probaba superioridad frente a otros equipos ni frente al ancla pública histórica0.924. No se ajustó la receta en respuesta al leaderboard.

## Incidencia N1
Una solicitud devolvió HTTPError sin ID confirmado. El gestor preservó la solicitud, pero guardó solo el tipo de error: se perdió el status y el mensaje del servidor. El historial no confirmó aceptación. Se decidió no reenviar para evitar duplicados y cerrar con CoAtNet único. Lección: guardar errores estructurados saneados y mantener una máquina de estados durable; nunca convertir incertidumbre en reintento ciego.

## Decisiones y límites
Priorizar una receta propia reproducible fue coherente con el alcance, pero no demostró que fuera la mejor estrategia competitiva. El gold pequeño y previamente utilizado limita la interpretación científica. El trabajo pendiente de investigación no se presenta como completado; no se publica una fusión, TTA o resultado clínico inexistente.
