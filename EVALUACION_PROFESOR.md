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

La web pública ejecuta ahora la CNN EMA y la fusión en el navegador mediante ONNX
Runtime Web, sin servidor Python. El profesor puede seleccionar las tres fases
PRE/EARLY/LATE compatibles, comprobar sus cortes y obtener la predicción real.
En fusión puede añadir edad y HR o permitir su imputación original. No necesita
cuenta ni instalación. Las imágenes y variables clínicas permanecen en su dispositivo.

La primera ejecución descarga los modelos. Se admiten PNG grises de 8 bits,
256 × 256, con igual número de cortes por fase (1–64). No se convierte una imagen
arbitraria al formato del estudio. Consultar [instrucciones y equivalencia](docs/browser-inference.md).
Esta equivalencia técnica no demuestra generalización a la validación privada;
no deben ajustarse pesos ni transformadores con esos pacientes.

## Datos y licencias

`evaluation/` contiene metadatos desidentificados del caso público y derivados de
investigación revisados contra esa fuente. Nunca debe añadirse el conjunto privado
de defensa. Las imágenes y el conjunto completo de BreastDCEDL se obtienen por
los enlaces de [acceso a datos](docs/original-data-access.md).

El código conserva CC BY 4.0; los datos y derivados tienen condiciones separadas
de uso académico y no comercial: [redistribución](docs/redistribution.md).
No se han realizado nuevos entrenamientos ni cambiado los resultados científicos.
