# Inferencia gratuita en el navegador

La web de GitHub Pages ejecuta la CNN EMA y la fusión final mediante ONNX Runtime
Web en CPU/WebAssembly. No necesita un servidor Python ni una suscripción.
El diseño y los resultados del estudio se conservan.

## Uso

1. Abrir la web en un navegador actualizado y entrar en **IA en directo** para la
   CNN, o **IA multimodal** para elegir imagen sola o imagen más clínica.
2. Seleccionar PRE, EARLY y LATE del mismo paciente: entre 1 y 64 cortes por fase,
   el mismo número en cada fase y PNG gris de 8 bits, 256 × 256, sin entrelazado.
3. Comprobar la tabla de correspondencia, ordenada por nombre de archivo, y marcar
   la confirmación. En fusión, introducir edad y HR o dejarlos ausentes.
4. Pulsar ejecutar. La primera ejecución descarga el motor y los modelos; después
   el cálculo se realiza en el dispositivo. **Las imágenes y variables clínicas
   no se envían a un servidor** ni se guardan en almacenamiento persistente.

No se admite una fotografía cualquiera, DICOM, PNG RGB, otra resolución o fases
desparejadas. No hay conversión, redimensionado ni reconstrucción automática.
Límites: 2 MiB por archivo y 64 MiB por selección. Los errores se muestran sin
inventar una predicción. Cambiar entradas, modelo o umbral invalida el resultado;
limpiar retira archivos y resultados de la interfaz.

## Modelos y equivalencia

Se exportan tres grafos sin reentrenar ni cuantizar: encoder CNN EMA, clasificador
original y cabeza de fusión. Se conservan los pesos, el orden PRE/EARLY/LATE,
normalizadores, media por paciente y transformador clínico originales. Los assets
se identifican por SHA-256 en `web/inference-assets/asset-manifest.json`.

Las pruebas de lectura de 30 PNG y normalización float32 coinciden exactamente con
Python; también la codificación clínica con valores presentes o ausentes. La
comparación PyTorch/ONNX y su tolerancia están en
[model-verification.json](../web/inference-assets/model-verification.json).
Las pruebas reales de navegador cubren CNN con 1/10 cortes y fusión de 10 cortes
con edad/HR presentes y ausentes; error máximo de probabilidad inferior a 3e-8.
Esto acredita equivalencia técnica en los casos comprobados, no validación clínica
ni equivalencia bit a bit en todos los dispositivos.

La AUC histórica 0,5524 corresponde a la CNN EMA en otro protocolo. La cabeza final
multimodal no tiene una evaluación independiente; no se le asigna la AUC OOF 0,6326.
La salida es una puntuación de investigación, no una probabilidad clínica calibrada.

## Alojamiento y límites

Pages sirve HTML y recursos estáticos de inferencia revisados. Los cómputos usan
CPU y memoria del visitante; la velocidad depende de su equipo y navegador.
No hacen falta GPU, cuenta de Hugging Face ni cuotas de inferencia. Los pesos son
descargables, igual que en el repositorio público; no se utiliza una clave secreta.

ONNX Runtime Web 1.31.1 se obtiene del paquete oficial npm con integridad SHA-512
verificada. Se conservan MIT y avisos de terceros. Las imágenes de pacientes y los
archivos OOF no se incluyen en el artefacto de Pages. No se han modificado los
originales ni realizado entrenamientos para preparar esta ejecución.
