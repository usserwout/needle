import concurrent.futures
import hashlib
import importlib.metadata
import inspect
import json
import os
import pickle
import platform
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# The jax metal plugin refuses to load against a newer PJRT API without this;
# it must be in the environment before jax initialises its backend, and this
# module is imported before any jax import on every training path.
if sys.platform == "darwin":
    os.environ.setdefault("ENABLE_PJRT_COMPATIBILITY", "1")

import numpy as np

from .tokenizer import (
    get_tokenizer, BOS_ID, EOS_ID, PAD_ID,
    IM_START, IM_END, THINK_START, THINK_END,
    TOOLS_START, TOOLS_END, TOOL_CALL_START, TOOL_CALL_END,
)

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "gate_proj", "out_proj")
DEFAULT_BASE = "checkpoints/needle2.pkl"

OPENROUTER_URL = os.environ.get(
    "OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"
)
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"

_GEN_SYSTEM = (
    "You generate training data for a tool-calling and extraction model. Given a "
    "set of tool or record schemas, produce realistic, diverse inputs paired with "
    "the exact calls that satisfy them. Return only JSON."
)

_GEN_TEMPLATE = """Schemas available (JSON):
{tools}

Produce {n} varied examples as a JSON array. Each element is an object:
  {{"query": "<a natural user request to act on, or a passage of text to extract from>",
    "reasoning": "<one short line deriving each argument from its source span in the query>",
    "answers": [{{"name": "<schema name>", "arguments": {{...}}}}]}}

Rules:
- Use only the schemas above; arguments must match them exactly and contain only
  values evidenced in the query.
- For an action tool the query is a command. For a record/extraction schema (its
  fields describe an entity), the query is a natural passage that contains those
  fields and the call extracts them.
- Cover single-call, multi-call, and about {refusals} off-topic inputs that no
  schema can serve (for those, "answers" is []).
- Vary phrasing, values, and which schemas are used. Return ONLY the JSON array."""


def _openrouter(messages, model, api_key, temperature=0.9):
    payload = json.dumps({"model": model, "messages": messages,
                          "temperature": temperature}).encode("utf-8")
    request = urllib.request.Request(OPENROUTER_URL, data=payload, headers={
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/cactus-compute/needle",
        "X-Title": "needle",
    })
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"]


def _parse_array(text):
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        rows = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [r for r in rows if isinstance(r, dict) and "query" in r and "answers" in r]


def generate_examples(tools, n=25, model=DEFAULT_MODEL, api_key=None, refusals=3):
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("set OPENROUTER_API_KEY to generate data")
    tools_json = tools if isinstance(tools, str) else json.dumps(tools, indent=2)
    prompt = _GEN_TEMPLATE.format(tools=tools_json, n=n, refusals=refusals)
    text = _openrouter([{"role": "system", "content": _GEN_SYSTEM},
                        {"role": "user", "content": prompt}], model, api_key)
    rows = _parse_array(text)
    for row in rows:
        row.setdefault("tools", tools if isinstance(tools, list) else json.loads(tools))
    return rows


def _dedup_key(example):
    answers = example.get("answers", example.get("function_calls", []))
    return (example.get("query", "").strip().lower(), json.dumps(answers, sort_keys=True))


def generate_dataset(tools, num_samples, model=DEFAULT_MODEL, batch_size=25,
                     api_key=None, workers=8, progress=None):
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("set OPENROUTER_API_KEY to generate data")

    target = int(num_samples * 1.3)
    max_submissions = max(1, target // batch_size * 3)
    seen, rows, failed, submitted = set(), [], 0, 0
    pool = ThreadPoolExecutor(max_workers=workers)
    pending = set()

    def _submit():
        nonlocal submitted
        pending.add(pool.submit(generate_examples, tools, batch_size,
                                model=model, api_key=api_key))
        submitted += 1

    for _ in range(min(workers, max(1, -(-target // batch_size)))):
        _submit()

    try:
        while pending and len(rows) < num_samples:
            done, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                try:
                    for example in future.result():
                        key = _dedup_key(example)
                        if key not in seen:
                            seen.add(key)
                            rows.append(example)
                except Exception as exc:
                    failed += 1
                    print(f"  {'failed':<9} {exc}", flush=True)
                if len(rows) < num_samples and submitted < max_submissions:
                    _submit()
            done_count = min(len(rows), num_samples)
            if progress:
                progress(done_count, num_samples)
            else:
                print(f"  {'generated':<9} {done_count}/{num_samples}  failed {failed}", flush=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return rows[:num_samples]


def _collect_tools(examples):
    seen, tools = set(), []
    for example in examples:
        for tool in example.get("tools", []):
            name = tool.get("name")
            if name and name not in seen:
                seen.add(name)
                tools.append(tool)
    return tools


def augment_jsonl(path, num_samples, model=DEFAULT_MODEL, batch_size=25, out_path=None, workers=8):
    with open(path) as handle:
        examples = [json.loads(line) for line in handle if line.strip()]
    tools = _collect_tools(examples)
    if not tools:
        raise RuntimeError("no tool schemas found in " + path)
    out_path = out_path or path.replace(".jsonl", "") + ".augmented.jsonl"
    generated = generate_dataset(tools, num_samples, model=model or DEFAULT_MODEL,
                                 batch_size=batch_size, workers=workers)
    with open(out_path, "w") as handle:
        for example in examples + generated:
            handle.write(json.dumps(example) + "\n")
    print(f"  {'wrote':<9} {len(examples) + len(generated)} examples  {out_path}")
    return out_path


def generate_main(args):
    model = args.model or DEFAULT_MODEL
    workers = getattr(args, "workers", 8)
    if args.tools:
        with open(args.tools) as handle:
            tools = json.load(handle)
        out = args.output or "needle_data.jsonl"
        rows = generate_dataset(tools, args.num_samples, model=model,
                                batch_size=args.batch_size, workers=workers)
        with open(out, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(f"  {'wrote':<9} {len(rows)} examples  {out}")
    elif args.augment:
        augment_jsonl(args.augment, args.num_samples, model=model,
                      batch_size=args.batch_size, out_path=args.output, workers=workers)
    else:
        raise SystemExit("pass --tools <schemas.json> or --augment <data.jsonl>")


def render_example(example):
    tools = example.get("tools", [])
    tools_json = tools if isinstance(tools, str) else json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
    answers = example.get("answers", example.get("function_calls", []))
    answers_json = answers if isinstance(answers, str) else json.dumps(answers, separators=(",", ":"), ensure_ascii=False)
    reasoning = (example.get("reasoning") or "").strip()
    system = (example.get("system") or "").strip()
    prefix = IM_START + "system\n" + system + IM_END + "\n" if system else ""
    prompt = (prefix + IM_START + "user\n" + TOOLS_START + tools_json + TOOLS_END + "\n"
              + example["query"] + IM_END + "\n" + IM_START + "assistant\n")
    think = THINK_START + "\n" + reasoning + "\n" + THINK_END + "\n" if reasoning else ""
    target = think + TOOL_CALL_START + answers_json + TOOL_CALL_END + IM_END
    return prompt, target


def _encode(tokenizer, example, max_len):
    prompt, target = render_example(example)
    prompt_ids = tokenizer.encode(prompt)
    target_ids = tokenizer.encode(target)
    ids = [BOS_ID] + prompt_ids + target_ids + [EOS_ID]
    mask = [0.0] * (1 + len(prompt_ids)) + [1.0] * (len(target_ids) + 1)
    ids, mask = ids[:max_len], mask[:max_len]
    pad = max_len - len(ids)
    return ids + [PAD_ID] * pad, mask + [0.0] * pad


def _rendered_lengths(path, tokenizer):
    lengths = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            if "query" not in example:
                continue
            prompt, target = render_example(example)
            lengths.append(len(tokenizer.encode(prompt)) + len(tokenizer.encode(target)) + 2)
    return lengths


def rendered_length_report(path, cap, tokenizer=None):
    """Return deployment-relevant rendered token statistics without truncating."""
    tokenizer = tokenizer or get_tokenizer()
    lengths = _rendered_lengths(path, tokenizer)
    return {
        "path": str(path), "examples": len(lengths), "max_len": max(lengths, default=0),
        "mean_len": float(np.mean(lengths)) if lengths else 0.0,
        "over_limit": sum(length > cap for length in lengths), "cap": cap,
    }


def fit_max_len(path, tokenizer, cap):
    longest = 0
    for length in _rendered_lengths(path, tokenizer):
        longest = max(longest, length)
    bucket = 128
    while bucket < min(longest, cap):
        bucket *= 2
    return min(bucket, cap)


def load_jsonl(path, tokenizer, max_len):
    seqs, masks = [], []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            if "query" not in example:
                continue
            ids, mask = _encode(tokenizer, example, max_len)
            seqs.append(ids)
            masks.append(mask)
    return np.array(seqs, np.int32), np.array(masks, np.float32)


def lora_target_paths(params):
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict
    flat = flatten_dict(params)
    paths = [path for path in flat
             if path[-1] == "kernel"
             and "stack" in path
             and "layers" in path
             and any(t in path for t in LORA_TARGETS)]
    paths = [p for p in paths if jnp.max(jnp.abs(flat[p])) > 1e-6]
    return paths


def init_lora(params, paths, rank, key):
    import jax
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict
    flat = flatten_dict(params)
    lora = {}
    for path in paths:
        weight = flat[path]
        in_dim, out_dim = weight.shape[-2], weight.shape[-1]
        lead = weight.shape[:-2]
        key, sub = jax.random.split(key)
        lora[path] = {
            "A": jax.random.normal(sub, lead + (in_dim, rank), jnp.float32) / rank,
            "B": jnp.zeros(lead + (rank, out_dim), jnp.float32),
        }
    return lora


def merge_lora(params, lora, scale):
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict, unflatten_dict
    flat = dict(flatten_dict(params))
    for path, adapter in lora.items():
        flat[path] = flat[path] + (scale * jnp.matmul(adapter["A"], adapter["B"])).astype(flat[path].dtype)
    return unflatten_dict(flat)


def _sha256(path):
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions():
    names = ("cactus-needle", "jax", "jaxlib", "flax", "optax", "numpy")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _serialize_lora(lora):
    return {"/".join(path): {"A": np.asarray(value["A"]), "B": np.asarray(value["B"])}
            for path, value in lora.items()}


def _restore_lora(serialized):
    import jax.numpy as jnp
    return {tuple(key.split("/")): {"A": jnp.asarray(value["A"]), "B": jnp.asarray(value["B"])}
            for key, value in serialized.items()}


def _write_adapter(path, lora, scale, base_path, rank, metadata):
    with open(path, "wb") as handle:
        pickle.dump({
            "lora": _serialize_lora(lora), "scale": float(scale), "base": base_path,
            "rank": rank, "metadata": metadata,
        }, handle)


def _patch_flax_debug_info(jax):
    """Bridge Flax's legacy debug_info call to newer JAX releases.

    JAX 0.4.38 (the version required by jax-metal) moved ``debug_info`` out of
    the public ``jax.api_util`` module and changed its signature. Flax 0.10.x
    still calls the old four-argument API from ``axes_scan``. Keep this shim
    local to fine-tuning so importing or serving Needle remains untouched.
    """
    if not hasattr(jax.api_util, "debug_info"):
        try:
            from jax._src import api_util as private_api_util
        except ImportError:
            private_api_util = None
        if private_api_util is not None:
            legacy = private_api_util.debug_info

            def compat(traced_for, fun, args, kwargs):
                try:
                    signature = inspect.signature(fun)
                except (TypeError, ValueError):
                    signature = None
                return legacy(traced_for, None, signature, args, kwargs, (), ())

            jax.api_util.debug_info = compat

    # The same JAX release removed the ``debug_info=`` keyword accepted by
    # Flax's ``linear_util.wrap_init`` call.  Preserve the old keyword and
    # discard it; debug metadata is non-functional for the training result.
    try:
        from jax.extend import linear_util
        if "debug_info" not in inspect.signature(linear_util.wrap_init).parameters:
            original_wrap_init = linear_util.wrap_init

            def wrap_init_compat(fun, params=None, **kwargs):
                return original_wrap_init(fun, params)

            linear_util.wrap_init = wrap_init_compat
    except (ImportError, TypeError, ValueError):
        pass


def finetune_local(args, progress=None):
    import jax
    import jax.numpy as jnp
    import optax
    _patch_flax_debug_info(jax)
    from .run import load_checkpoint
    from .architecture import SimpleAttentionNetwork

    def emit(msg):
        print(msg, flush=True)
        if progress:
            progress(msg)

    base_path = args.checkpoint or DEFAULT_BASE
    data_path = getattr(args, "train_file", None) or args.jsonl_path
    val_path = getattr(args, "val_file", None)
    seed = getattr(args, "seed", 0)
    if getattr(args, "generate", 0):
        data_path = augment_jsonl(data_path, args.generate, model=getattr(args, "model", None),
                                  workers=getattr(args, "workers", 8))
        if val_path:
            emit("  note     generated examples are added to train data only")

    params, config = load_checkpoint(base_path)
    config.dtype = "float32"
    params = jax.tree.map(lambda a: np.asarray(a).astype(np.float32), params)
    backend = jax.default_backend().lower()
    if backend == "metal":
        # Manual GQA attention is the conservative Metal path.  Some newer
        # jax-metal builds can lower dot_product_attention, so expose an
        # opt-in for benchmarking without making it the default stable path.
        config.flash = os.environ.get("NEEDLE_METAL_FLASH", "0") == "1"
        config.remat = False
        config.scan_unroll = config.num_layers
    params = jax.device_put(params)
    emit(f"  {'backend':<9} {backend}  float32")
    tokenizer = get_tokenizer(config.vocab_size)
    cap = args.max_len
    length_paths = [data_path] + ([val_path] if val_path else [])
    length_reports = [rendered_length_report(path, cap, tokenizer) for path in length_paths]
    over_limit = sum(report["over_limit"] for report in length_reports)
    for report in length_reports:
        emit(f"  {'tokens':<9} {os.path.basename(report['path'])}  max {report['max_len']} "
             f"mean {report['mean_len']:.1f}  over {report['over_limit']}/{report['cap']}")
    if over_limit and getattr(args, "fail_on_truncation", False):
        raise SystemExit(f"{over_limit} examples exceed --max-len {cap}; shorten or remove them")
    max_len = max(fit_max_len(path, tokenizer, cap) for path in length_paths)
    seqs, masks = load_jsonl(data_path, tokenizer, max_len)
    if len(seqs) == 0:
        raise SystemExit("no usable examples in " + data_path)
    emit(f"  {'data':<9} {len(seqs)} examples  seq_len {max_len}  cap {args.max_len}")

    model = SimpleAttentionNetwork(config)
    paths = lora_target_paths(params)
    scale = args.lora_alpha / args.lora_rank
    lora = init_lora(params, paths, args.lora_rank, jax.random.PRNGKey(seed))
    emit(f"  {'lora':<9} rank {args.lora_rank}  alpha {args.lora_alpha:g}  {len(paths)} weight groups")

    if val_path:
        val_seqs, val_masks = load_jsonl(val_path, tokenizer, max_len)
        n_val = len(val_seqs)
        emit(f"  {'validation':<9} {n_val} explicit grouped examples")
    else:
        n_val = min(int(len(seqs) * getattr(args, "val_split", 0.1)), len(seqs) - 1)
        if n_val > 0:
            order = np.random.default_rng(seed).permutation(len(seqs))
            seqs, masks = seqs[order], masks[order]
            val_seqs, val_masks = seqs[:n_val], masks[:n_val]
            seqs, masks = seqs[n_val:], masks[n_val:]
            emit(f"  {'holdout':<9} {n_val} examples for validation (prefer --val-file)")
        else:
            val_seqs = val_masks = None

    batch, count = args.batch_size, len(seqs)
    accum_steps = max(1, getattr(args, "grad_accum_steps", 1))
    micro_steps_per_epoch = -(-count // batch)
    steps_per_epoch = -(-micro_steps_per_epoch // accum_steps)
    requested_steps = getattr(args, "max_steps", 0)
    total_steps = requested_steps or args.epochs * steps_per_epoch
    epoch_limit = (max(args.epochs, -(-total_steps // steps_per_epoch))
                   if requested_steps else args.epochs)
    warmup = min(max(1, total_steps // 20), max(1, total_steps - 1))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr,
        warmup_steps=warmup, decay_steps=max(total_steps, warmup + 1))
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule))
    opt_state = optimizer.init(lora)
    emit(f"  {'schedule':<9} {total_steps} updates  warmup {warmup}  cosine decay  clip 1.0 "
         f"microbatch {batch} x accumulation {accum_steps}  (compiling...)")

    def loss_sum_fn(lora, ids, mask):
        logits = model.apply({"params": merge_lora(params, lora, scale)}, ids)
        logits, targets, mask = logits[:, :-1], ids[:, 1:], mask[:, 1:]
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return (ce * mask).sum(), mask.sum()

    @jax.jit
    def grad_step(lora, ids, mask):
        (loss_sum, token_count), grads = jax.value_and_grad(loss_sum_fn, has_aux=True)(lora, ids, mask)
        return grads, loss_sum, token_count

    @jax.jit
    def apply_step(lora, opt_state, grads):
        updates, opt_state = optimizer.update(grads, opt_state, lora)
        return optax.apply_updates(lora, updates), opt_state

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    metadata = {
        "train_sha256": _sha256(data_path), "val_sha256": _sha256(val_path),
        "base_sha256": _sha256(base_path), "seed": seed, "backend": backend,
        "platform": platform.platform(), "git_revision": _git_revision(),
        "package_versions": _package_versions(),
        "config": {"lr": args.lr, "lora_rank": args.lora_rank,
                   "lora_alpha": args.lora_alpha, "max_len": max_len,
                   "microbatch": batch, "grad_accum_steps": accum_steps},
    }
    resume_signature = {
        "train_sha256": metadata["train_sha256"], "val_sha256": metadata["val_sha256"],
        "base_sha256": metadata["base_sha256"], "lr": args.lr,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "max_len": max_len, "microbatch": batch, "grad_accum_steps": accum_steps,
        "selection_metric": getattr(args, "selection_metric", "val_loss"),
    }
    start_epoch = 0
    resume_start = 0
    step_i = 0
    best_val = float("inf")
    best_score = float("-inf")
    best_lora = _serialize_lora(lora)
    no_improvement = 0
    resume_path = getattr(args, "resume", None)
    if resume_path:
        with open(resume_path, "rb") as handle:
            state = pickle.load(handle)
        if state.get("base") != base_path or state.get("rank") != args.lora_rank:
            raise ValueError("resume state does not match base checkpoint or LoRA rank")
        if state.get("resume_signature") != resume_signature:
            raise ValueError("resume state does not match the dataset or optimizer configuration")
        lora = _restore_lora(state["lora"])
        opt_state = jax.tree.map(jnp.asarray, state["opt_state"])
        start_epoch = int(state.get("next_epoch", int(state.get("epoch", -1)) + 1))
        resume_start = int(state.get("next_start", 0))
        step_i = int(state.get("step", 0))
        best_val = float(state.get("best_val", best_val))
        best_score = float(state.get("best_score", -best_val))
        best_lora = state.get("best_lora", best_lora)
        no_improvement = int(state.get("no_improvement", 0))
        emit(f"  {'resume':<9} {resume_path}  step {step_i}  epoch {start_epoch + 1} "
             f"offset {resume_start}")

    eval_loss_sum = jax.jit(loss_sum_fn)
    eval_every = getattr(args, "eval_every", 0) or steps_per_epoch
    patience = getattr(args, "early_stopping_patience", 0)
    selection_metric = getattr(args, "selection_metric", "val_loss")

    def runtime_exact_score():
        """Score a temporary export through deployed constrained decoding."""
        from .architecture import effective_kv_window
        from .export import write_export
        from needle.dutch import evaluate_runtime

        archive = os.path.join(args.checkpoint_dir, f".needle_runtime_eval_{step_i}.cact")
        merged = merge_lora(params, lora, scale)
        write_export(merged, config, archive, bits=4, tokenizer=tokenizer,
                     kv_window=effective_kv_window(config))
        try:
            report = evaluate_runtime(val_path, weights=archive, bootstrap_samples=0, seed=seed)
            return report["exact_call_accuracy"], report
        finally:
            try:
                os.remove(archive)
            except FileNotFoundError:
                pass

    def evaluate_and_checkpoint(next_epoch, next_start, last_loss):
        nonlocal best_val, best_score, best_lora, no_improvement
        val = None
        runtime = None
        if n_val > 0:
            total_loss, total_tokens = 0.0, 0.0
            for start in range(0, n_val, batch):
                loss_sum, token_count = eval_loss_sum(
                    lora, jnp.asarray(val_seqs[start:start + batch]),
                    jnp.asarray(val_masks[start:start + batch]))
                total_loss += float(loss_sum)
                total_tokens += float(token_count)
            val = total_loss / max(total_tokens, 1.0)
            if selection_metric == "runtime_exact":
                if not val_path:
                    raise ValueError("--selection-metric runtime_exact requires --val-file")
                score, runtime = runtime_exact_score()
            else:
                score = -val
            if score > best_score:
                best_val, best_score, best_lora, no_improvement = val, score, _serialize_lora(lora), 0
                _write_adapter(os.path.join(args.checkpoint_dir, "needle_lora_best.pkl"), lora,
                               scale, base_path, args.lora_rank,
                               {**metadata, "step": step_i, "validation_loss": val,
                                "selection_metric": selection_metric, "selection_score": score,
                                "runtime_evaluation": runtime})
            else:
                no_improvement += 1
        _write_adapter(os.path.join(args.checkpoint_dir, f"needle_lora_step_{step_i}.pkl"), lora,
                       scale, base_path, args.lora_rank,
                       {**metadata, "step": step_i, "validation_loss": val})
        state_path = os.path.join(args.checkpoint_dir, "needle_training_state.pkl")
        with open(state_path, "wb") as handle:
            pickle.dump({
                "lora": _serialize_lora(lora), "opt_state": jax.tree.map(np.asarray, opt_state),
                "base": base_path, "rank": args.lora_rank,
                "next_epoch": next_epoch, "next_start": next_start, "step": step_i,
                "best_val": best_val, "best_score": best_score, "best_lora": best_lora,
                "no_improvement": no_improvement, "resume_signature": resume_signature,
            }, handle)
        suffix = f"  val {val:.4f}" if val is not None else ""
        if runtime is not None:
            suffix += f"  exact {runtime['exact_call_accuracy']:.4f}"
        emit(f"  {'checkpoint':<9} next epoch {next_epoch + 1}  step {step_i}  loss {last_loss:.4f}{suffix}")
        return bool(patience and n_val and no_improvement >= patience)

    should_stop = False
    last = 0.0
    for epoch in range(start_epoch, epoch_limit):
        order = np.random.default_rng(seed + epoch).permutation(count)
        grad_sum = None
        token_sum = 0.0
        micro_count = 0
        checkpoint_epoch, checkpoint_start = epoch, resume_start if epoch == start_epoch else 0
        for start in range(checkpoint_start, count, batch):
            idx = order[start:start + batch]
            grads, loss_sum, token_count = grad_step(lora, jnp.asarray(seqs[idx]), jnp.asarray(masks[idx]))
            grad_sum = grads if grad_sum is None else jax.tree.map(lambda left, right: left + right, grad_sum, grads)
            token_sum += float(token_count)
            micro_count += 1
            if micro_count < accum_steps and start + batch < count:
                continue
            grads = jax.tree.map(lambda value: value / max(token_sum, 1.0), grad_sum)
            lora, opt_state = apply_step(lora, opt_state, grads)
            last = float(loss_sum) / max(float(token_count), 1.0)
            step_i += 1
            grad_sum, token_sum, micro_count = None, 0.0, 0
            checkpoint_epoch, checkpoint_start = (epoch + 1, 0) if start + batch >= count else (epoch, start + batch)
            if step_i % eval_every == 0:
                should_stop = evaluate_and_checkpoint(checkpoint_epoch, checkpoint_start, last)
            if step_i >= total_steps or should_stop:
                break
        if step_i % eval_every != 0:
            should_stop = evaluate_and_checkpoint(checkpoint_epoch, checkpoint_start, last) or should_stop
        if should_stop or step_i >= total_steps:
            break
        resume_start = 0

    out = args.out or os.path.join(args.checkpoint_dir, "needle_lora.pkl")
    selected_lora = _restore_lora(best_lora) if n_val else lora
    _write_adapter(out, selected_lora, scale, base_path, args.lora_rank,
                   {**metadata, "step": step_i, "validation_loss": best_val if n_val else None,
                    "selected_by": selection_metric if n_val else "final_step",
                    "selection_score": best_score if n_val else None})
    print(f"  {'adapter':<9} {out}")
    print(f"  {'next':<9} needle build {base_path} --lora {out}")
    print(f"  {'note':<9} confidence reports None with tuned weights; the head is not tuned")


def build_main(args):
    import jax.numpy as jnp
    from .run import load_checkpoint
    from .architecture import effective_kv_window
    from .export import write_export

    params, config, _ = load_checkpoint(args.checkpoint, return_run=True)

    if args.lora:
        with open(args.lora, "rb") as handle:
            adapter = pickle.load(handle)
        lora = {tuple(key.split("/")): {"A": jnp.asarray(v["A"]), "B": jnp.asarray(v["B"])}
                for key, v in adapter["lora"].items()}
        params = merge_lora(params, lora, adapter["scale"])
        print(f"  {'merged':<9} {len(lora)} weight groups  {args.lora}")

    bits = args.bits
    bits_map = None if bits else (getattr(config, "weight_bits", "") or None)
    if not bits and not bits_map:
        bits = "4"

    out = args.out or (os.path.splitext(os.path.basename(args.checkpoint))[0] + ".cact")
    info = write_export(params, config, out,
                        bits=int(bits) if bits else 4,
                        bits_map=bits_map,
                        tokenizer=get_tokenizer(config.vocab_size),
                        kv_window=effective_kv_window(config))
    scheme = f"mixed[{bits_map}]" if bits_map else f"W{bits}"
    print(f"  {'wrote':<9} {info['path']}  {info['bytes'] / 1e6:.2f} MB  {info['tensors']} tensors  {scheme}A8")
    print(f"  {'next':<9} needle.Needle(weights={out!r}, tools=[...])")

    if args.upload:
        repo = os.environ.get("NEEDLE_HF_REPO")
        if not repo:
            raise SystemExit("set NEEDLE_HF_REPO=<you>/<model> to upload")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo, repo_type="model", exist_ok=True)
        api.upload_file(path_or_fileobj=out, path_in_repo=os.path.basename(out),
                        repo_id=repo, repo_type="model")
        print(f"  {'uploaded':<9} {os.path.basename(out)}  {repo}")
