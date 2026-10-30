# Entrega pública actual
+
+El repositorio contiene 17 modelos conservados, transformadores, código, metadatos
+públicos, particiones, predicciones, métricas y presentación UAX en PPTX/PDF.
+No se han reentrenado modelos ni cambiado los resultados originales.
+
+La web terminada ejecuta ahora CNN EMA y fusión final **en el navegador**, sin
+backend Python ni cuota mensual. Las imágenes y variables clínicas permanecen en
+el dispositivo del visitante. La descarga inicial incluye el runtime y los grafos
+ONNX revisados; el diseño y las curvas del estudio se conservan.
+
+Comprobaciones: 28 tests; encoder/clasificador/fusión ONNX comparados con PyTorch;
+preprocesamiento PNG y clínico exacto; cuatro casos reales de navegador con error
+de probabilidad máximo inferior a 3e-8; entradas incompatibles rechazadas y retirada
+de resultados obsoletos. Se comprobó el ancho móvil de 390 píxeles sin desbordamiento.
+Véanse `browser-inference.md` y `browser-verification.json`.
+
+Las MRI se obtienen aparte. No se publica la validación privada del profesor ni
+datos introducidos por visitantes. No hay pesos guardados de fusión lineal ni
+clínico final global; la cabeza multimodal final sigue sin validación independiente.
+El acceso público funcional no elimina estas limitaciones científicas.
+
+Los originales y respaldos permanecen intactos. Datos, código, runtime y elementos
+institucionales conservan sus condiciones separadas de licencia y atribución.
+