# Needle in Electron without native compilation

## Recommendation

Use the official WebAssembly build in an Electron utility process. Keep the renderer sandboxed and expose a small IPC API such as `complete`, `reset`, and `extract` through the preload script.

This avoids `node-gyp`, Electron ABI rebuilds, and separate addon packages. It also keeps synchronous inference off the main and renderer threads. Ship three files as application resources:

- `needle.js`, the Emscripten CommonJS loader
- `needle.wasm`, the inference engine
- `needle2.cact`, the model and tokenizer

The native command-line runner remains a useful alternative when maximum CPU performance matters. It requires a different executable for each operating system and architecture, while the same WASM files work across Electron targets.

## Verified facts

The official release lists `needle.js` and `needle.wasm` for Browser and Node deployments. It also publishes standalone executables for supported desktop targets. The runner supports a long-lived HTTP mode on localhost with `POST /complete`, plus a one-query command-line mode. See the [Needle 2 model card](https://huggingface.co/Cactus-Compute/needle2/blob/main/README.md).

The official C header exports `needle_load`, `needle_init`, `needle_complete`, and `needle_reset`. See [`needle.h`](https://huggingface.co/Cactus-Compute/needle2/blob/main/wasm/needle.h).

The current Emscripten loader detects Node outside Electron renderer processes. In Node it reads `needle.wasm` from the directory containing `needle.js`. It exports the four Needle functions together with `_malloc`, `_free`, `HEAPU8`, and `UTF8ToString`. See the official [`needle.js`](https://huggingface.co/Cactus-Compute/needle2/resolve/main/wasm/needle.js).

Electron's `utilityProcess.fork()` starts a Node process and supplies message-port IPC. It can only be started after the Electron app is ready. See Electron's [`utilityProcess` API](https://www.electronjs.org/docs/latest/api/utility-process).

Packagers normally put application code in a read-only ASAR archive. Electron Builder's `extraResources` copies binary and data assets outside ASAR into the application resources directory. Runtime code can locate that directory through `process.resourcesPath`. See [Electron Builder application contents](https://www.electron.build/docs/contents/) and Electron's [`process.resourcesPath`](https://www.electronjs.org/docs/latest/api/process#processresourcespath-readonly).

## Local verification

Verified on 2026-08-31 with the repository's installed Node runtime:

- `needle.js`: 62,433 bytes
- `needle.wasm`: 333,324 bytes
- `needle2.cact`: 13,737,807 bytes
- WASM module creation: successful with no compilation
- Model load: return code `0`
- Tool initialization: successful
- Invoice extraction: successful in 1,384 ms on the first measured completion
- Extracted value: `{"company":"Acme Corp","total":1200.5}`

This is a compatibility check, not a benchmark. Production measurements should include warm runs and each supported CPU architecture.

## Suggested Electron layout

```text
resources/needle/
  needle.js
  needle.wasm
  needle2.cact
src/main/
  needle-process.cjs
  needle-client.ts
src/preload/
  index.ts
```

Example Electron Builder configuration:

```yaml
extraResources:
  - from: vendor/needle
    to: needle
    filter:
      - needle.js
      - needle.wasm
      - needle2.cact
```

The utility process should load the module once, copy the `.cact` bytes into WASM memory once, and serialize all calls. `needle_complete` is synchronous, so the process must not accept a second inference request until the current one finishes.

## API details that are easy to get wrong

- Pass the `.cact` byte length to `_needle_load` as a JavaScript `BigInt` because the C parameter is an unsigned 64-bit integer.
- `_needle_load` uses `0` for success.
- `_needle_init` and `_needle_complete` return non-negative values on success. Only a negative value indicates failure.
- Allocate NUL-terminated UTF-8 strings in WASM memory.
- Free temporary input and output allocations after each call.
- Keep one reusable output buffer if completions run serially.
- The engine owns process-global model and conversation state. Use one utility process per concurrently active independent engine session.
- The current wrapper rejects `toolIndexPath`; an index path from Node's host filesystem is not automatically visible inside Emscripten's virtual filesystem. Add an explicit file mount before exposing this optimization.
- Extraction is a one-tool call. Initialize the engine with the extraction JSON Schema, call `complete`, and return the first call's `arguments` after application-side validation.

## Native runner alternative

The standalone runner also needs no compilation. An Electron main process can ship it through `extraResources`, start it with `--tools tools.json --serve`, and call its local `/complete` endpoint.

I would choose this only after measuring WASM and finding it too slow. The runner adds per-platform packaging, executable signing, lifecycle handling, port collision handling, and a local HTTP endpoint. The official release currently lists macOS only for Apple Silicon, while WASM does not have that architecture gap.
