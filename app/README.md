# Cosmian Confidential AI

API to run a language model in a confidential VM

## Install dependencies

By default all dependencies will be installed with the app.
If you don't need CUDA support, you can save space by installing PyTorch for CPU only:

```sh
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

To use Intel AVX/AMX extensions:

```sh
pip install intel-extension-for-pytorch
```

## Build and install the app

```sh
python -m build
pip install dist/*.whl
```

## Test it

```sh
CONFIG_PATH="./tests/config.json" python tests/test.py
```

## Serve the API

Be sure to have `~/.local/bin` in your `PATH`

```sh
CONFIG_PATH="./tests/config.json" cosmian-ai-runner
```

## Config file

The config is written as a JSON file, with different parts:

### Auth

Optional part to fill the identity providers information.
If no `auth` information is present in the config file, the authentication will be disabled.

It should contain a list of the following fields per identity providers:

- `jwks_uri`: identity provider's JSON Web Key Set

- `client_id`: ID of the client calling this application set by the identity provider

### Summary

Information about the summarization models to use and generation parameters.

Different models can be used depending on the language of the document to summarize.
It is **mandatory** to have at least a `default` model entry.

We recommend to use `facebook/bart-large-cnn` (400M parameters).
You can specify a custom `generation_config`, you can see the default one for your model on HuggingFace: [generation_config.json](https://huggingface.co/facebook/bart-large-cnn/blob/main/generation_config.json).
You can find more information about text generation [here](https://huggingface.co/blog/how-to-generate).

### Translation

Information about the translation model to use and generation parameters.

We recommend to use `facebook/nllb-200-distilled-600M` (600M parameters).

### Models

Models configuration defines available models for `/predict` and `/rag` routes.

For classic huggingface pipeline models, model configuration should be structured as follow:

```json
        {
            "model_id": "facebook/bart-large-cnn",
            "task": "summarization",
            "prompt": "{text}",
            "kwargs": {
                "temperature": 0.1,
                "do_sample": true,
                "truncation": true,
                "max_new_tokens": 200
            }
        }
```

For gguf models, model should be structured as follow (selected file, over different available quantizations must be precised):

```json
      {
          "model_id": "TheBloke/Spring-Dragon-GGUF",
          "file": "spring-dragon.Q2_K.gguf",
          "prompt": "",
          "kwargs": {
              "max_new_tokens": 256,
              "temperature": 0.01,
              "context_length": 4096,
              "repetition_penalty": 1.1,
              "gpu_layers": 0
          }
      }
```

### Sentence Transformer

Sentence transformer configuration defines which setence transformer to use when creating RAG (on server launch).

```json
  {
    "sentence_transformer":
    {
        "file": "sentence-transformers/all-MiniLM-L12-v2",
        "score_threshold": 0.12
    }
  }
  ```

If no element is configured, RAG is not created and `/rag` routes will send an error when fetched.

### Sample config file

```json
{
  "auth": {
    "openid_configs": [
      {
        "client_id": "XXXX",
        "jwks_uri": "XXXX"
      }
    ]
  },
  "summary": {
    "default": {
      "model_name": "facebook/bart-large-cnn",
      "generation_config": {
        "max_length": 140,
        "min_length": 30
      }
    }
  },
  "translation": {
    "model_name": "facebook/nllb-200-distilled-600M",
    "generation_config": {
      "max_length": 200
    }
  }
}
```

## API Endpoints

### Summarize text

- Endpoint: `/summarize`
- Method: **POST**
- Description: get the summary of a given text, using the configured model
- Request:
  - Headers: 'Content-Type: multipart/form-data'
  - Body: `doc` - text to summarize, using model configured on summary config section
- Response:
```json
  {
    "summary": "summarized text..."
  }
```
- Example:
```
curl 'http://0.0.0.0:5000/summarize' \
--form 'doc="Il était une fois, dans un royaume couvert de vert émeraude et voilé dans les secrets murmurants des arbres anciens, vivait une princesse nommée Elara.."'
```

### Translate text

- Endpoint: `/translate`
- Method: **POST**
- Description: get the translation of a given text, using model configured on translation config section
- Request:
  - Headers: 'Content-Type: multipart/form-data'
  - Body:
    `doc` - text to translate
    `src_lang` - source language
    `tgt_lang` - targeted language
- Response:
```json
  {
    "translation": "translated text..."
  }
```
- Example:
```
curl 'http://0.0.0.0:5000/translate' \
--form 'doc="Il était une fois, dans un royaume couvert de vert émeraude et voilé dans les secrets murmurants des arbres anciens, vivait une princesse nommée Elara.."' --form 'src_lang=fr'  --form 'tgt_lang=en'
```

### Predict

- Endpoint: `/predict`
- Method: **POST**
- Description: get prediction from a model available in application configuration (using
  HuggingFacePipeline, or a gguf model)
- Request:
  - Headers: 'Content-Type: multipart/form-data'
  - Body:
    `text` - text to use for prediction
    `model` - model_id to use for inference
- Response:
```json
  {
    "response": "generated text..."
  }
```
- Example:
```
curl 'http://0.0.0.0:5000/predict' \
--form 'text="What color is a banana?"' --form 'model="google/flan-t5-small"'
```

You can list available models from current configuration using:
- Endpoint: `/models`
- Method: **GET**

###  RAG

Those routes are available if a sentence_transformer has been provided in the configuration file.
The sentence_transformer is used to create and infer vectors
- Endpoint: `/rag`
- Method: **POST**
- Description: get prediction from a RAG, using a model available in application configuration
- Request:
  - Headers: 'Content-Type: multipart/form-data'
  - Body:
    `text` - text to use for rag request
    `model` - model_id to use for inference after RAG step
- Response:
```json
  {
    "response": "generated text..."
  }
```
- Example:
```
curl 'http://0.0.0.0:5000/rag' \
--form 'text="Who is Esmeralda?"' --form 'model="facebook/bart-large-cnn"'
```

You can add an `.epub` document to the vector DB of the current RAG, using:
- Endpoint: `/add_document`
- Method: **POST**
- File sent on multipart
- Example:
```
curl -F "file=@/path/data/Victor_Hugo_Notre-Dame_De_Paris_en.epub" http://0.0.0.0:5000/add_document
```

*So far, only epub files can be handled.*
