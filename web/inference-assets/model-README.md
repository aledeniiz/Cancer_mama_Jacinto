# Modelos ONNX para inferencia local en el navegador

Conversión de la CNN EMA y cabeza multimodal final conservadas en `models/trained/`.
Sin reentrenamiento, cuantización, sustitución de modelos ni cambios de resultados.
La CNN recibe PRE/EARLY/LATE estandarizados y produce 64 rasgos por corte. Se
promedian todos los cortes del paciente y se aplica el clasificador original.
La fusión recibe ese promedio normalizado y las cuatro variables clínicas
codificadas con el transformador original; usa su propia cabeza conservada.

`model-config.json` incluye contratos, transformadores y hashes de origen.
`model-verification.json` compara PyTorch y ONNX Runtime CPU, incluyendo un paciente
del caso público cuando se proporcionan MRI locales. Ninguna MRI se distribuye.
Las tolerancias no acreditan una validación externa ni utilidad clínica. La cabeza
multimodal final no hereda la AUC OOF de las cabezas por fold.

Exportación opcional: `python scripts/export_browser_models.py --output NUEVO_DIR`.
Para repetir la prueba desde imágenes añadir `--image-root RAIZ_PNG` y un
`--evidence` externo nuevo. Los archivos de salida existentes nunca se sobrescriben.
Dependencias de exportación: onnx 1.23.2, onnxruntime 1.31.1 y el runtime del proyecto.
