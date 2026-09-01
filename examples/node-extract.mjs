import path from "node:path";
import { fileURLToPath } from "node:url";
import { Needle } from "../node/needle.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const engineDir = process.env.NEEDLE_ENGINE_DIR ?? path.join(here, "..", "vendor", "needle");

const invoice = {
  name: "invoice",
  description: "The issuing company and final payable invoice total.",
  parameters: {
    type: "object",
    properties: {
      company: { type: "string", description: "Company that issued the invoice" },
      total: { type: "number", minimum: 0, description: "Final total in major currency units" },
    },
    required: ["company", "total"],
  },
};

const agent = await Needle.create({ engineDir, tools: [invoice] });
const result = await agent.extract(
  "Invoice from Acme Corp. Subtotal $1,100.00. Total: $1,200.50.",
  invoice,
);

console.log(JSON.stringify(result, null, 2));
