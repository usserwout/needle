import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const textEncoder = new TextEncoder();
const extractionAgents = new Map();

/**
 * Load the official Needle WebAssembly engine and model.
 *
 * `engineDir` must contain needle.js and needle.wasm. `modelPath` defaults to
 * needle2.cact in that same directory. The files are deliberately supplied
 * by the application so Electron can package them as extraResources.
 */
export class Needle {
  static async create({
    engineDir,
    modelPath = null,
    tools = [],
    system = "",
    toolIndexPath = null,
    outputCapacity = 1 << 20,
  }) {
    if (!engineDir) {
      throw new TypeError("Needle.create requires engineDir");
    }
    if (toolIndexPath) {
      throw new Error("toolIndexPath is not supported by the current WASM wrapper");
    }

    const loaderPath = path.join(engineDir, "needle.js");
    const wasmPath = path.join(engineDir, "needle.wasm");
    const resolvedModelPath = modelPath ?? path.join(engineDir, "needle2.cact");
    const missing = [];
    for (const assetPath of [loaderPath, wasmPath, resolvedModelPath]) {
      try {
        await fs.access(assetPath);
      } catch {
        missing.push(assetPath);
      }
    }
    if (missing.length) {
      throw new Error(
        `Needle assets are missing. Expected needle.js, needle.wasm, and needle2.cact in ${engineDir}. `
        + `Run examples/download-assets.mjs or set NEEDLE_ENGINE_DIR. Missing: ${missing.join(", ")}`,
      );
    }
    const [createNeedle, wasmBinary, cact] = await Promise.all([
      Promise.resolve(require(loaderPath)),
      fs.readFile(wasmPath),
      fs.readFile(resolvedModelPath),
    ]);

    const module = await createNeedle({ wasmBinary });
    const instance = new Needle(module, { outputCapacity });
    instance._loadModel(cact);
    const schemas = tools.map((tool) => {
      if (tool && typeof tool.execute === "function") {
        instance.addTool(tool);
        const { execute, ...schema } = tool;
        return schema;
      }
      return tool;
    });
    instance.schemas = schemas;
    instance._init({ tools: schemas, system, toolIndexPath });
    return instance;
  }

  constructor(module, { outputCapacity = 1 << 20 } = {}) {
    this.module = module;
    this.outputCapacity = outputCapacity;
    this.handlers = new Map();
    this.queue = Promise.resolve();
  }

  /** Register a function for run(). The schema is passed to the engine. */
  addTool({ name, description, parameters, execute }) {
    if (typeof execute !== "function") {
      throw new TypeError(`Tool ${name} needs an execute function`);
    }
    this.handlers.set(name, execute);
    return { name, description, parameters };
  }

  complete(input, maxNewTokens = 256) {
    return this._serial(() => this._complete(input, maxNewTokens));
  }

  async run(input, { maxSteps = 8, maxNewTokens = 256 } = {}) {
    let response = await this.complete(input, maxNewTokens);
    const results = [];

    for (let step = 0; step < maxSteps; step += 1) {
      const calls = response.function_calls ?? [];
      if (response.type !== "call" || calls.length === 0) break;

      const stepResults = [];
      for (const call of calls) {
        const execute = this.handlers.get(call.name);
        if (!execute) {
          stepResults.push({ error: `unknown tool: ${String(call.name)}` });
          continue;
        }
        try {
          stepResults.push(await execute(call.arguments ?? {}));
        } catch (error) {
          stepResults.push({ error: error instanceof Error ? error.message : String(error) });
        }
      }
      results.push(...stepResults);
      response = await this.complete(serializeToolResults(stepResults), maxNewTokens);
    }

    return { ...response, results };
  }

  async extract(input, schema, maxNewTokens = 256) {
    if (this.schemas?.length !== 1 || this.schemas[0]?.name !== schema?.name) {
      throw new Error("extract() requires an agent configured with exactly this one schema");
    }
    const response = await this.complete(input, maxNewTokens);
    const call = response.function_calls?.[0];
    return call ? call.arguments : null;
  }

  reset() {
    return this._serial(() => {
      this.module._needle_reset();
    });
  }

  _loadModel(cact) {
    const pointer = this.module._malloc(cact.byteLength);
    try {
      this.module.HEAPU8.set(cact, pointer);
      const rc = this.module._needle_load(pointer, BigInt(cact.byteLength));
      if (rc !== 0) throw new Error(`needle_load failed (code ${rc})`);
    } finally {
      this.module._free(pointer);
    }
  }

  _init({ tools, system, toolIndexPath }) {
    const systemPointer = this._stringPointer(system ?? "");
    const toolsPointer = this._stringPointer(JSON.stringify(tools));
    const indexPointer = toolIndexPath ? this._stringPointer(toolIndexPath) : 0;
    try {
      const rc = this.module._needle_init(systemPointer, toolsPointer, indexPointer);
      if (rc < 0) throw new Error(`needle_init failed (code ${rc})`);
    } finally {
      this.module._free(systemPointer);
      this.module._free(toolsPointer);
      if (indexPointer) this.module._free(indexPointer);
    }
  }

  _complete(input, maxNewTokens) {
    const inputPointer = this._stringPointer(input);
    const outputPointer = this.module._malloc(this.outputCapacity);
    try {
      const rc = this.module._needle_complete(
        inputPointer,
        Number(maxNewTokens),
        outputPointer,
        this.outputCapacity,
      );
      if (rc < 0) throw new Error(`needle_complete failed (code ${rc})`);
      return JSON.parse(this.module.UTF8ToString(outputPointer));
    } finally {
      this.module._free(inputPointer);
      this.module._free(outputPointer);
    }
  }

  _stringPointer(value) {
    const bytes = textEncoder.encode(`${value}\0`);
    const pointer = this.module._malloc(bytes.byteLength);
    this.module.HEAPU8.set(bytes, pointer);
    return pointer;
  }

  _serial(operation) {
    const result = this.queue.then(operation, operation);
    this.queue = result.catch(() => undefined);
    return result;
  }
}

/** Return a schema plus its executable handler for use with Needle.run(). */
export function defineTool(tool) {
  if (!tool || typeof tool.name !== "string") {
    throw new TypeError("defineTool requires a tool name");
  }
  if (typeof tool.execute !== "function") {
    throw new TypeError(`Tool ${tool.name} needs an execute function`);
  }
  return tool;
}

/** Create a one-tool extraction agent and return its arguments. */
export async function extract(input, schema, options) {
  const createOptions = options ?? {};
  const key = JSON.stringify({
    engineDir: createOptions.engineDir,
    modelPath: createOptions.modelPath ?? null,
    system: createOptions.system ?? "",
    schema,
  });
  let agent = extractionAgents.get(key);
  if (!agent) {
    agent = await Needle.create({ ...createOptions, tools: [schema] });
    extractionAgents.set(key, agent);
    while (extractionAgents.size > 8) {
      extractionAgents.delete(extractionAgents.keys().next().value);
    }
  } else {
    extractionAgents.delete(key);
    extractionAgents.set(key, agent);
  }
  await agent.reset();
  return agent.extract(input, schema, options?.maxNewTokens ?? 256);
}

function serializeToolResults(results) {
  try {
    return JSON.stringify(results, (_key, value) =>
      typeof value === "bigint" ? value.toString() : value);
  } catch (error) {
    return JSON.stringify([{ error: `tool result was not JSON serializable: ${String(error)}` }]);
  }
}
