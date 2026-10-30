# GitHub Pages e inferencia local

URL pública: https://aledeniiz.github.io/Cancer_mama_Jacinto/

El workflow de `.github/workflows/pages.yml` despliega desde `main` mediante GitHub
Actions. La web conserva su diseño, capítulos, curvas y comparación científica.
Además ejecuta CNN EMA y fusión final en el navegador con ONNX Runtime Web.
No despliega Python ni necesita un servicio de cómputo de pago.

El artefacto contiene exclusivamente el HTML revisado, `.nojekyll` y la lista
cerrada de recursos de `inference-assets/`: tres ONNX, configuración original,
runtime WASM/JS, procedencia, licencias y manifiesto. `build_pages.py` comprueba
hashes y tamaños antes de copiar. No incluye MRI, pacientes, OOF, código Git ni
el resto del repositorio. El runtime y los modelos se descargan al preparar la
primera inferencia; las entradas del visitante nunca se envían al servidor.

```sh
python -B scripts/audit_release.py
python -B web/build_public.py --check
python -B -m unittest discover -s tests -p test_pages.py
python -B scripts/build_pages.py --output .pages-site
```

El destino de empaquetado debe ser un directorio nuevo. La publicación se verifica
con el workflow y los bytes servidos por HTTPS, además de la ejecución real en el
navegador. Véanse [uso y pruebas](browser-inference.md) y
[documentación oficial de Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
