# Modelos conservados y recargables

Los pesos y transformadores revisados están en [trained/](trained/), con su informe
de procedencia, hashes y pruebas de conversión. No se han reentrenado ni optimizado.

- CNN EMA histórica, 24.713 parámetros: TEST histórico de 176 pacientes, AUC 0,5524.
- Cinco CNN por fold: evaluadores OOF distintos de la EMA global.
- Cinco regresiones logísticas clínicas: coeficientes, interceptos recuperados,
  imputación, escalado y codificación originales. No son un modelo clínico global.
- Cinco cabezas de fusión neuronal y una cabeza final de desarrollo, con sus
  transformadores y CNN asociada. La cabeza final no tiene validación independiente.
- Fusión lineal: se conservan sus predicciones OOF; los parámetros que no fueron
  serializados se documentan, sin fabricar ni entrenar un sustituto.

Los cargadores públicos no deserializan pickle. Las CNN y cabezas usan directorios
JSON/NPZ con `numpy.load(allow_pickle=False)`. Los `.pt` originales permanecen
intactos fuera del repositorio; la conversión conserva sus tensores. Los clínicos
usan JSON con variables y transformadores explícitos.

Las referencias están en [evaluation/](../evaluation/). Los embeddings permiten
revisar etapas posteriores al encoder sin imágenes; para inferencia desde píxeles
se necesitan las resonancias y el preprocesamiento original. No se equipara la
comprobación de referencias con una nueva validación independiente.

El runtime probado utiliza PyTorch 2.10.0; el entrenamiento histórico usó 2.5.1.
Las verificaciones distinguen tensores exactamente conservados de las tolerancias
numéricas de inferencia. No se promete repetición bit a bit del entrenamiento GPU.

Para ejecutar CNN EMA y fusión sin Python, la web incorpora grafos ONNX preservados;
consulta [inferencia en navegador](../docs/browser-inference.md).
