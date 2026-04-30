# Native Source Inventory

These folders contain source files copied from the sibling research/library directories used by the notebooks.

The notebooks historically made these modules importable with:

```python
sys.path.append(r"D:\Dropbox\TRADING\TRADING PROPERTY LIBRARIES")
```

Current folders:

- `goertzel`: Goertzel extension source from `GOERTZEL TRANSFORM C/V1.2`.
- `cyfitness`: C++ fitness evaluator source.
- `cygaopt`: C++ genetic optimization source and CMake file.
- `cygaopt_multicore`: multicore C++ genetic optimization source and CMake file.
- `genetic_optimization_legacy`: older C genetic optimization source currently commented out in `cyPredict/__init__.py`.

Generated files such as `.pyd`, `build/`, Visual Studio, and CMake outputs are intentionally ignored by Git.

Build with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_native.ps1
```

The build script defaults to the Anaconda environment interpreter:

```text
C:\Users\Federico\anaconda3\envs\cyenv\python.exe
```

The default build output is each module's `build/lib.*` directory. cyPredict
prefers those build directories at import time, then falls back to in-place
`.pyd` files. Use `-InPlace` only after closing Python processes that may have
loaded the previous native modules.
