# Datos, unidades y procedencia

`evaluation/` distribuye metadatos desidentificados del caso público y sus derivados
revisados: etiquetas, particiones, predicciones y embeddings. Se han comparado
con la descarga pública del caso para excluir la validación privada del profesor.
No se redistribuyen imágenes de pacientes.

| Unidad | Subconjunto preparado | Desarrollo | TEST histórico |
|---|---:|---:|---:|
| Pacientes | 1.273 | 1.097 | 176 |
| Tripletes PRE/EARLY/LATE | 12.703 | 10.945 | 1.758 |
| Archivos PNG de fase | 38.109 | 32.835 | 5.274 |

La comparación clínica/fusión usa 1.092 pacientes de desarrollo, excluyendo cinco
resultados desconocidos. Las particiones son por paciente, no por imagen. El
conjunto completo publicado por BreastDCEDL contiene 2.070 pacientes y es distinto
del subconjunto docente preparado.

Los campos clínicos usados son edad y HR basales; HR es numérico 0/1. Los valores
ausentes se tratan con transformadores ajustados únicamente en entrenamiento.
No se usan etiquetas ni variables postratamiento como predictores. La auditoría
de fecha de adquisición individual es incompleta y sigue siendo una limitación.

Las imágenes deben descargarse según [el acceso original](original-data-access.md).
Las rutas PRE/EARLY/LATE en metadatos son referencias relativas, no imágenes
incluidas. Ningún script de comprobación descarga datos automáticamente.

La fuente es Fridman et al., Scientific Data 13, 264 (2026),
[DOI](https://doi.org/10.1038/s41597-026-06589-6).
[Zenodo 18114231](https://zenodo.org/records/18114231) declara CC BY-NC 4.0.
Conservar su atribución y las condiciones de las cohortes originales. El código
tiene licencia separada; véase [redistribución](redistribution.md).
