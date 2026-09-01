# Dutch training-data source

## Verified facts

- The official MASSIVE repository publishes a tarball containing one JSONL file per locale; the archive layout includes `1.0/data/nl-NL.jsonl` and `1.0/data/en-US.jsonl`. See the [official MASSIVE README](https://github.com/alexa/massive/blob/main/README.md).
- The Hugging Face dataset card identifies the Dutch configuration as `nl-NL`, exposes `train`, `dev`, and `test` partitions, and documents the fields used by the converter (`utt`, `annot_utt`, `intent`, `scenario`, and `partition`). See the [official dataset card](https://huggingface.co/datasets/AmazonScience/massive).
- The dataset is released under CC BY 4.0; preserve attribution and the source hash in the generated manifest.

## Recommendation

Use the official tarball for this repository's first run because it already provides the JSONL format expected by `needle dutch-data`. The Hugging Face loader is a valid alternative when you want programmatic filtering, but it requires an additional conversion step.
