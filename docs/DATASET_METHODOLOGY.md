# Dataset methodology

The manifest records the official/source repositories, split, and target size (50). To avoid an unexpected large download, preparation does not auto-download data. Download an official source after reviewing its dataset card and licence, then run `python -m benchmarks.prepare_datasets gsm8k path\\to\\official.jsonl --limit 50`. The process preserves source fields in metadata and writes a timestamped provenance manifest. No generated replacement examples are used.
