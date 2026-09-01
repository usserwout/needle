import path from "node:path";
import { fileURLToPath } from "node:url";
import { Needle, defineTool } from "../node/needle.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const engineDir = process.env.NEEDLE_ENGINE_DIR ?? path.join(here, "..", "vendor", "needle");

const weather = defineTool({
  name: "get_weather",
  description: "Get the current weather for a city.",
  parameters: {
    type: "object",
    properties: { city: { type: "string", description: "City to look up" } },
    required: ["city"],
  },
  async execute({ city }) {
    // Replace this with your application or device integration.
    return { city, temperature_c: 21, sky: "clear" };
  },
});

const agent = await Needle.create({
  engineDir,
  tools: [weather],
});

const response = await agent.run("What's the weather in Brussels?");
console.log(JSON.stringify(response, null, 2));
