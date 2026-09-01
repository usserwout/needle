import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const outputDir = process.env.NEEDLE_ENGINE_DIR ?? path.join(here, "..", "vendor", "needle");
const baseUrl = "https://huggingface.co/Cactus-Compute/needle2/resolve/main";
const assets = ["wasm/needle.js", "wasm/needle.wasm", "needle2.cact"];

await fs.mkdir(outputDir, { recursive: true });
for (const asset of assets) {
  const response = await fetch(`${baseUrl}/${asset}`);
  if (!response.ok) throw new Error(`Could not download ${asset}: HTTP ${response.status}`);
  const destination = path.join(outputDir, path.basename(asset));
  await fs.writeFile(destination, Buffer.from(await response.arrayBuffer()));
  console.log(`Downloaded ${destination}`);
}
