# Evaluación y reproducción del trabajo

La web terminada está en [GitHub Pages](https://aledeniiz.github.io/Cancer_mama_Jacinto/).
El repositorio incorpora los artefactos científicos conservados, además del código.
Las imágenes de resonancia se descargan por separado; no se incluyen entornos
virtuales, cachés, respaldos ni datos privados de evaluación del profesor.

## Comprobar resultados sin descargar resonancias

Con Python 3.11, instalar las dependencias siguiendo el README y ejecutar:

```sh
python -B scripts/reproduce_results.py
python -B -m pytest -q -p no:cacheprovider
```

El primer comando comprueba los resultados a partir de las predicciones originales.
Consulte su salida y los informes de `evaluation/` y `models/trained/` para las
comprobaciones de identificadores, particiones, transformadores y recarga de pesos.
Verificar predicciones guardadas no equivale a realizar una validación externa.
Recalcular toda la cadena desde píxeles requiere las resonancias originales.

## Qué se compara

| Protocolo | Modelo | Pacientes | ROC-AUC |
|---|---|---:|---:|
| OOF emparejado, cinco folds | CNN por fold | 1.092 | 0,5197 |
| OOF emparejado, cinco folds | Solo clínica, edad y HR | 1.092 | 0,6450 |
| OOF emparejado, cinco folds | CNN + clínica, fusión neuronal | 1.092 | 0,6326 |
| OOF emparejado, cinco folds | Fusión lineal | 1.092 | 0,6355 |
| TEST histórico, otro protocolo | CNN EMA | 176 | 0,5524 |

La CNN EMA de 0,5524 no sustituye a la CNN por fold en una comparación OOF.
La fusión neuronal mejora numéricamente la CNN aislada, pero no demuestra
una ventaja sobre el modelo clínico. Los cinco modelos clínicos OOF no son
un modelo final entrenado con todos los pacientes. La cabeza multimodal final
tampoco dispone de una estimación independiente de rendimiento.

## Demostración con pacientes nuevos

GitHub Pages es una presentación estática: no ejecuta Python ni permite inferencia
real con los archivos de un paciente. Para la prueba práctica de la guía docente
se necesita ejecutar el pipeline de inferencia con los pesos y el preprocesamiento
originales en un entorno con backend. La existencia de esta web pública no acredita
por sí sola ese requisito de despliegue. Los comandos locales y sus contratos se
describen en `examples/README.md`; no se deben reemplazar las predicciones por
simulaciones ni ajustar el modelo con los casos privados del profesor.

## Datos y licencias

`evaluation/` contiene metadatos desidentificados del caso público y derivados de
investigación revisados contra esa fuente. Nunca debe añadirse el conjunto privado
de defensa. Las imágenes y el conjunto completo de BreastDCEDL se obtienen por
los enlaces de [acceso a datos](docs/original-data-access.md).

El código conserva CC BY 4.0; los datos y derivados tienen condiciones separadas
de uso académico y no comercial: [redistribución](docs/redistribution.md).
No se han realizado nuevos entrenamientos ni cambiado los resultados científicos.
