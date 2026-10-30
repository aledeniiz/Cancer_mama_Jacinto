# Verificación de la entrega científica

Fecha: 30 de octubre de 2026. Sin entrenamientos, optimización ni cambios de
arquitectura o resultados originales. Se comprobó en Python 3.11 y PyTorch
2.10.0 CPU; las versiones están en [tested-environment.json](tested-environment.json).

## Comprobaciones realizadas

- **25 tests superados**, incluidos los controles originales, el artefacto
  estático de Pages, integridad de la lista permitida y reproducción científica.
- **17 modelos recargados en 17 procesos independientes**; sus parámetros conservan
  192 tensores exactamente iguales a los originales. Los cinco clínicos preservan
  coeficientes, interceptos recuperados y transformadores. Véanse el
  [registro](../models/trained/registry.json) y la
  [recarga](../models/trained/reload_verification.json).
- Metadatos del caso público comparados byte a byte con su descarga: 1.273 pacientes
  y 12.703 tripletes; etiquetas, edad y HR coincidentes. La validación privada del
  profesor no está incluida. La licencia y atribución están en `evaluation/`.
- Cobertura y orden original de 1.092 predicciones OOF, cinco folds por paciente,
  particiones de desarrollo/TEST y 176 predicciones del TEST histórico comprobados.
- ROC-AUC y BCE originales recalculados desde los scores guardados. Se respeta
  el tipo original float32 de la fusión al reconstruir la aritmética de sus métricas;
  convertir indiscriminadamente todo a float64 cambia los últimos decimales de BCE.
- Predicciones de clínica, clasificador CNN y fusión neuronal recalculadas desde
  sus parámetros y los embeddings originales. Error máximo de probabilidad:
  clínica **2.2204460492503131e-16**, CNN por fold **8.9406967218597089e-08**, fusión **0**.
  La fusión se compara tras restaurar el float32 original: las probabilidades
  coinciden exactamente; los decimales de serialización del CSV no son float64 originales.
  Las tolerancias y diferencias de métricas se informan explícitamente en la salida.
- 19 archivos críticos del proyecto original comparados con sus hashes previos:
  todos intactos. No se alteraron originales ni respaldos.

## Alcance real

La comprobación del encoder desde píxeles necesita las MRI, que se descargan por
separado. El replay de embeddings no sustituye esa prueba ni una validación externa.
Las pruebas artificiales de recarga verifican software, no rendimiento científico.
No se repite el entrenamiento histórico GPU ni se promete equivalencia bit a bit.

Los cinco clínicos son modelos OOF: no existe un clínico global final guardado.
La fusión lineal no guardó su estimador; se preservan sus predicciones y métricas,
sin fabricar un sustituto. La cabeza multimodal final no tiene una evaluación
independiente. No se mezcla su resultado con la CNN EMA del TEST histórico.

## Web, entrega y revisión

Se conserva el diseño de la plataforma y la selección visible de tres modelos
multimodales. El navegador verificó los avisos nuevos, seis cargas desactivadas
y ausencia de errores de consola durante esa comprobación. El generador comprueba
que el HTML coincide con sus fuentes. Pages sigue sin ejecutar inferencia Python.
La presentación final conserva sus cinco diapositivas y sus archivos originales.

El escáner sigue revisando todos los archivos, incluso ignorados, excepto el
almacenamiento interno de Git. Los formatos de datos/modelos y sus identificadores
solo se permiten en rutas científicas revisadas con hash y tamaño exactos. Un
archivo añadido o cambiado fuera de esa lista queda bloqueado. Las comprobaciones
de credenciales y rutas de máquina siguen aplicándose. El escaneo es evidencia
limitada, no una prueba universal de ausencia de secretos.

## Repetir

```sh
python -B scripts/reproduce_results.py
python -B models/trained/verify_reload.py
python -B -m pytest -q -p no:cacheprovider
python -B scripts/audit_release.py
python -B web/build_public.py --check
```

Los manifests científicos y de modelos se validan antes del replay. La lista
`scientific_artifact_allowlist.json` delimita los artefactos aprobados y
`release_manifest.json` recoge los archivos entregados, excluyéndose a sí mismo.
Cambiar esa lista exige revisar la procedencia, licencia y hashes; no basta con
desactivar el escáner. El diagnóstico vacío local ignorado no forma parte de la entrega.
