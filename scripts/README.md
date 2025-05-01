# Benchmark and interact with the API

## Sample data

- `sample_en_doc.txt`: sample english documentation from Cloudproof (~7000 characters)

- `sample_fr_doc.txt`: extract of a French tale (~4000 characters)

## Clients

Use the API to perform summarize or translation.

See [./client](./client) for instructions.

## Benchmark

### `bench_inference.py`

Directly measure inference time on the machine it is executed.

```bash
python bench_inference.py sample_data/sample_en_doc.txt [-n 2] [--verbose]
```
