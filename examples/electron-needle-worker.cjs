// Run this file with Electron's utilityProcess.fork(), not in a renderer.
let agent;
let queue = Promise.resolve();

async function handle(message) {
  if (message.type === "init") {
    const { Needle } = await import("../node/needle.mjs");
    agent = await Needle.create(message.options);
    return { ready: true };
  }
  if (!agent) throw new Error("Needle worker has not been initialized");
  if (message.type === "complete") {
    return agent.complete(message.input, message.maxNewTokens ?? 256);
  }
  if (message.type === "run") {
    return agent.run(message.input, message.options ?? {});
  }
  if (message.type === "extract") {
    return agent.extract(message.input, message.schema, message.maxNewTokens ?? 256);
  }
  if (message.type === "reset") {
    await agent.reset();
    return null;
  }
  throw new Error(`Unknown Needle worker message: ${message.type}`);
}

process.parentPort.on("message", (event) => {
  const message = event.data ?? event;
  queue = queue.then(async () => {
    try {
      process.parentPort.postMessage({ id: message.id, ok: true, value: await handle(message) });
    } catch (error) {
      process.parentPort.postMessage({
        id: message.id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });
});
