# Plataforma pública e inferencia en navegador

Se conserva la web terminada: diseño, animaciones educativas, capítulos y
resultados originales. La comparación visual sigue mostrando sus tres entradas
seleccionadas; los resultados científicos completos se conservan en el repositorio.

IA en directo y el laboratorio multimodal ejecutan ahora CNN EMA y fusión final
en el navegador con los pesos y transformadores originales exportados a ONNX.
La lectura PNG no utiliza conversiones de color del canvas; valida resolución,
escala de grises, profundidad, CRC y tamaño antes del preprocesamiento original.
Los datos del usuario no se envían al servidor, no se registran ni persisten.

No se incluyen MRI ni se cargan registros OOF o pacientes en el explorador. Los
visuales educativos siguen identificados como sintéticos. La inferencia real se
distingue de esos ejemplos y sus errores nunca generan puntuaciones simuladas.
Los controles exigen fases emparejadas y confirmación; los cambios invalidan el
resultado anterior. La cabeza final de fusión no tiene validación independiente.

El generador incrusta el código propio; el runtime y tres modelos se sirven como
recursos estáticos propios revisados y se descargan al usarlos. No hay CDN, backend
Python, token ni API externa de inferencia. Los hashes y las licencias están en
`inference-assets/`. Ver `docs/browser-inference.md` para uso y comprobaciones.
