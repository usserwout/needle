# Node.js examples

These examples use the official WebAssembly runtime. Download or copy the `wasm` folder and `needle2.cact` from the [Needle 2 model release](https://huggingface.co/Cactus-Compute/needle2) into `vendor/needle`, or point `NEEDLE_ENGINE_DIR` at a directory containing:

```text
needle.js
needle.wasm
needle2.cact
```

From this directory, the asset download helper creates the expected layout automatically:

```sh
node download-assets.mjs
```

Run either example with Node 18 or newer:

```sh
NEEDLE_ENGINE_DIR=/path/to/needle-assets node examples/node-toolcalling.mjs
NEEDLE_ENGINE_DIR=/path/to/needle-assets node examples/node-extract.mjs
```

The wrapper is dependency-free. In an Electron app, load it from the main process or an Electron utility process and expose only the operations your preload API needs to the renderer.

Reuse one `Needle` instance for repeated work. The top-level `extract()` helper also keeps up to eight warm extraction agents, keyed by their model and schema.

`electron-main.cjs` and `electron-needle-worker.cjs` show the utility-process boundary. Add your real tool handlers in the worker or forward calls to the main process, depending on which side owns the device or application APIs.
