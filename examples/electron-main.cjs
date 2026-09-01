// Minimal main-process wiring. Keep the renderer behind preload IPC.
const path = require("node:path");
const { app, ipcMain, utilityProcess } = require("electron");

let child;
let nextId = 1;
const pending = new Map();

function request(type, payload = {}) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    child.postMessage({ id, type, ...payload });
  });
}

app.whenReady().then(async () => {
  child = utilityProcess.fork(path.join(__dirname, "electron-needle-worker.cjs"));
  child.on("message", (message) => {
    const waiter = pending.get(message.id);
    if (!waiter) return;
    pending.delete(message.id);
    if (message.ok) waiter.resolve(message.value);
    else waiter.reject(new Error(message.error));
  });

  const engineDir = path.join(process.resourcesPath, "needle");
  await request("init", {
    options: {
      engineDir,
      tools: [{
        name: "get_weather",
        description: "Get the current weather for a city.",
        parameters: {
          type: "object",
          properties: { city: { type: "string" } },
          required: ["city"],
        },
      }],
    },
  });

  ipcMain.handle("needle:complete", (_event, input) => request("complete", { input }));
  ipcMain.handle("needle:reset", () => request("reset"));
});
