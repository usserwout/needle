# Finetuning

Needle finetunes with LoRA adapters on the frozen base model: rank 16 on the five attention projections of every layer, trained on your JSONL, then merged into the weights at export. The engine, the tokenizer, and the confidence head are untouched. The output is a `.cact` archive you pass as `weights=`.

## Data format

Install the training dependencies before running the commands in this guide:

```sh
pip install "cactus-needle[train]"
```

One JSON object per line. `query` and `tools` describe the turn, `answers` lists the exact calls the model should emit, `reasoning` is one short line deriving each argument from its source span in the query. `reasoning` is optional but include it: the model produces the derivation before the call, and examples that show where each value comes from teach grounding, not just tool selection.

```json
{"query": "Bantilan, N. (2018). Themis. Journal of Technology in Human Services, 36(1).", "tools": [{"name": "extract_citation_data", "parameters": {"type": "object", "properties": {"authors": {"type": "string"}, "title": {"type": "string"}, "publisher": {"type": "string"}}, "required": ["authors", "title"]}}], "answers": [{"name": "extract_citation_data", "arguments": {"authors": "Bantilan, N.", "title": "Themis", "publisher": "Journal of Technology in Human Services, 36(1)."}}], "reasoning": "authors precede the year; title follows the year; publisher is the journal segment"}
```

Rules that matter:

1. Arguments contain only values present in the query. Omit optional fields with no evidence; never fill them with placeholders or empty strings.
2. Include off topic examples with `"answers": []`. The built in generator produces about 1 in 8. Without them the tuned model calls a tool on everything.
3. When the catalogue has similar tools, include ambiguous queries resolved to the correct one.
4. An optional `"system"` field per example becomes a system turn, matching `Needle(system=...)` at inference.
5. Each rendered example must fit within `--max-len`. Use `--fail-on-truncation` for a real run so a long target cannot be silently cut off. Padding adjusts to the longest example automatically, rounded up to a power of two. A short dataset trains much faster than a dataset padded to 1024 tokens.

## Dutch data workflow

Needle's fine-tuning target is a structured tool call, not a general chat reply. The Dutch workflow trains Dutch requests against English tool and field names, then evaluates actual engine calls on a held-out set.

Download the official MASSIVE `nl-NL.jsonl` and `en-US.jsonl` files yourself, then run one command to build the complete corpus. With the defaults below, the command reads `data/raw/1.0/data/nl-NL.jsonl` and `data/raw/1.0/data/en-US.jsonl`, writes to `data/dutch`, and includes 20,000 additional deterministic variation records. The builder never calls a hosted model and writes attribution and source hashes into `manifest.json`.

```sh
needle dutch-data
```

Override the paths or augmentation size when needed:

```sh
needle dutch-data \
  --input path/to/nl-NL.jsonl \
  --input-en path/to/en-US.jsonl \
  --output data/dutch \
  --augmentation-count 50000 \
  --seed 0
```

Use `--augmentation-count 0` for a base-only corpus.

The generated corpus contains balanced MASSIVE action requests, deterministic Dutch extraction records, no-call cases, multi-call cases, and English replay. `review.jsonl` lists every deterministic validation and test record that needs a human review before a release.

Validate before training. Start with `--max-len 256` because the deployed engine has a short working context. Do not train a corpus that reports invalid examples, duplicates across splits, or over-limit examples.

```sh
needle validate-data data/dutch/train.jsonl data/dutch/val.jsonl data/dutch/test.jsonl \
  --require-grounding --max-len 256 --fail-on-truncation \
  --report data/dutch/validation.json
```

### Add more deterministic Dutch variation

For a larger vocabulary and more paraphrase/register variation, append locally
generated examples to the training split. This generator uses fixed Dutch
lexicons and templates (standard Netherlands Dutch, Belgian Dutch, and noisy
registers), and it emits schema-valid labels with every string argument copied
from the query. It does not call Gemini or any hosted API.

```sh
needle dutch-augment \
  --input data/dutch/train.jsonl \
  --output data/dutch/train-augmented.jsonl \
  --count 20000 \
  --seed 0
```

Use `--count 50000` or more when local storage and training time allow. The
output is a new file; the original training split is preserved. Validate the
augmented file together with the untouched validation and test partitions:

```sh
needle validate-data data/dutch/train-augmented.jsonl data/dutch/val.jsonl data/dutch/test.jsonl \
  --require-grounding --max-len 256 --fail-on-truncation \
  --report data/dutch/augmented-validation.json
```

The mix is approximately 60% single-call actions, 20% extraction, 10% hard
no-call cases, and 10% multi-call requests. Keep the frozen `val.jsonl` and
`test.jsonl` unchanged.
The expanded action vocabulary includes robot navigation and object handling,
sentiment-analysis requests, gaming actions, lighting and temperature control,
travel booking, music, translation, and order/shipment operations. Extraction
schemas additionally cover receipts, expenses, shipping labels, and line items.

Run a short benchmark first on an Apple Silicon Mac. Begin with one-example microbatches and use gradient accumulation to reach an effective batch size of 16. The first JAX compile takes time, so measure at least 50 optimizer updates before estimating a complete run.

```sh
pip install "cactus-needle[train,metal]"
needle finetune data/dutch/train.jsonl \
  --val-file data/dutch/val.jsonl \
  --max-len 256 --fail-on-truncation \
  --batch-size 1 --grad-accum-steps 16 \
  --max-steps 50 --lora-rank 16 --lora-alpha 32 --lr 1e-4 \
  --checkpoint-dir checkpoints/dutch-benchmark
```

The trainer writes a resumable optimizer state, adapters after each validation, and `needle_lora_best.pkl`. Pass `--selection-metric runtime_exact` for the Dutch workflow. It exports each validation checkpoint temporarily and selects by real engine exact-call accuracy. Validation loss remains in the report as a diagnostic. Resume an interrupted run with `--resume checkpoints/dutch-benchmark/needle_training_state.pkl`.

For the pilot, run 800 optimizer updates at `1e-4` and `2e-4` in separate checkpoint directories. Train the full corpus with the better run, up to five epochs, and stop after two non-improving validations.

```sh
needle finetune data/dutch/train.jsonl \
  --val-file data/dutch/val.jsonl \
  --max-len 256 --fail-on-truncation \
  --batch-size 1 --grad-accum-steps 16 \
  --epochs 5 --early-stopping-patience 2 \
  --selection-metric runtime_exact \
  --lora-rank 16 --lora-alpha 32 --lr 1e-4 \
  --checkpoint-dir checkpoints/dutch-seed0 --seed 0
```

Export and score through the same engine that will run in production. Do this in separate processes when comparing base and tuned weights because the engine cannot unload tuned weights.

```sh
needle build checkpoints/needle2.pkl \
  --lora checkpoints/dutch-seed0/needle_lora.pkl \
  --bits 4 --out dutch-reference.cact
needle evaluate data/dutch/test.jsonl --weights dutch-reference.cact \
  --report reports/dutch-reference.json
needle build checkpoints/needle2.pkl \
  --lora checkpoints/dutch-seed0/needle_lora.pkl \
  --out dutch-compact.cact
needle evaluate data/dutch/test.jsonl --weights dutch-compact.cact \
  --report reports/dutch-compact.json
needle release-check --report reports/dutch-reference.json \
  --compact-report reports/dutch-compact.json
```

The evaluation report contains exact-call accuracy, a bootstrap confidence interval, tool selection, argument match, no-call F1, multi-call accuracy, hallucinated-argument rate, schema validity, and slice results. `release-check` enforces the Dutch accuracy, confidence interval, no-call, hallucination, slice, and quantization gates. Pass the English base and tuned reports plus the second-seed Dutch report when they are available.

## Commands

Train, export, load:

```sh
needle finetune data.jsonl --epochs 10 --out adapter.pkl
needle build checkpoints/needle2.pkl --lora adapter.pkl --out tuned.cact
```

```python
agent = needle.Needle(tools=[...], weights="tuned.cact")
```

To share a tuned model, set `NEEDLE_HF_REPO=<you>/<model>` and pass `--upload` to `needle build`; on any other machine `needle download <you>/<model>/tuned.cact` pulls it back down (or pass just `<you>/<model>` when the repo holds a single archive).

Defaults: batch size 16, learning rate 0.0001 with warmup and cosine decay, gradient clipping at norm 1, rank 16, alpha 32, max length 1024, validation split 0.1. The base checkpoint downloads from Hugging Face on first run.

For a production-quality run, pass `--val-file` instead of relying on `--val-split`. The latter is kept for small ad hoc experiments and may leak related templates into validation.

To grow a small hand written set, seed the generator with it (needs `OPENROUTER_API_KEY`; set `OPENROUTER_URL` to use another OpenAI compatible gateway):

```sh
needle generate-data --augment data.jsonl --num-samples 1000
```

The playground button labelled Finetune on these tools runs the same pipeline from the browser.

Training is plain JAX, so it runs on any accelerator jax supports. On an NVIDIA machine install the CUDA build and the same command trains on the GPU, nothing else changes:

```sh
pip install "cactus-needle[train,gpu]"
```

Apple GPUs train through the jax metal plugin, which does not work past jax 0.4.38, so the `metal` extra pins an older stack:

```sh
pip install "cactus-needle[train,metal]"
```

Needle detects the Metal backend and adapts automatically: manual attention, no rematerialisation, unrolled layer stack, and the `ENABLE_PJRT_COMPATIBILITY` variable the plugin requires is set for you. Measured on an M5 Max: 0.71 seconds per step against 2.90 on CPU at the same shape, about 4 times faster, with a one time compile of about 23 seconds. Training runs in float32 on every backend.

## Reading the loss

The loss covers only the target: the reasoning line plus the JSON call. Much of the call is boilerplate the base model already predicts (the tool name, the braces, the field names), so training starts near 1.0 rather than near random. Judge a run by its trend, not its level.

Step count is what small datasets get wrong. 200 examples at batch 16 is 13 steps per epoch, and the default 3 epochs is 39 steps total, which barely moves a rank 16 adapter at the default learning rate. For a few hundred examples run 10 to 30 epochs and expect a clear downward trend. If the curve sits at its starting value after a few hundred steps, raise the epochs first, then the learning rate.

A validation loss prints at each epoch end (10 percent of examples are held out by default, `--val-split` to change). When it rises while the training loss keeps falling, the run is overfitting: stop there, or add data.

Training also prints a progress line every 25 optimizer updates by default. It reports the update count and percentage, loss, updates per second, elapsed time, estimated time remaining, and distance to the next evaluation. The first update includes JAX compilation and is reported separately, so the ETA is only meaningful after compilation completes. Change the cadence with `--progress-every 10` (or a larger value for less output). Evaluation and checkpoint lines include their own duration and total elapsed time.

## Sizing the dataset

Tool selection moves first: a few hundred clean examples measurably improve which tool gets picked. Argument grounding moves later and needs more data, on the order of thousands of examples, with reasoning lines and varied phrasings and values. If evaluation shows correct tools with wrong argument values, the dataset is too small or too uniform, not mislabeled. For grounding heavy tasks `--lora-rank 32` doubles adapter capacity and the adapter stays tiny.

For a large catalogue, consider two passes at inference instead of more training: one turn against the full catalogue to pick the tool, then one turn declaring only that tool, which constrains the grammar to exactly that call.

## What finetuning does not change

The confidence head. Scores are calibrated for the base model on its training mix and finetuning does not update the head, so the package disables them for tuned weights: `Needle(weights=...)` warns once at construction and reports `confidence` as None. Non English deployments of the base model should also treat the score with caution (correct Spanish calls have been measured at confidence 0.0).

The tokenizer. Non English text fragments into roughly 1.7 times more tokens (measured on Spanish), which taxes both quality and the 256 token window.

## Troubleshooting

1. Loss goes NaN within the first steps on CPU: fixed, run `pip install --upgrade cactus-needle`.
2. The tuned model answers "Sorry, I can't help with that" on everything: an old engine gated low confidence responses; the gate is removed as of engine 2.0.1. Upgrade the package.
3. `failed to load weights`: the `.cact` format is tied to the engine version, so an archive exported by an older package will not load. Rebuild it with the current package version.
4. Loss hovers at its starting value: see Reading the loss. The run is undertrained, not broken.
