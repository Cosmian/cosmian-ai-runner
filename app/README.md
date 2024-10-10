# Cosmian Confidential AI

API to run a language model in a confidential VM

## Install dependencies

## CPU

By default all dependencies will be installed with the app.
If you don't need CUDA support, you can save space by installing PyTorch for CPU only:

```sh
cp requirements.cpu.txt requirements.txt
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

To use Intel AVX/AMX extensions:

```sh
pip install intel-extension-for-pytorch
```

### GPU Apple Silicon (Metal)

To use the Apple Silicon Metal GPUs, install the following requirements: [https://github.com/context-labs/mactop]

```shell
cp requirements.metal.txt requirements.txt
CT_METAL=1  pip install -r requirements.txt
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

### Databases

Databases configuration defines available models with associated sentence transformer for `/predict` route.

For classic huggingface pipeline models and associated sentence tranformer, databases configuration should be structured as follow:

```json
    "databases": [
        {
            "name": "Litterature",
            "model": {
                "model_id": "facebook/bart-large-cnn",
                "task": "summarization",
                "prompt": "{text}",
                "kwargs": {
                    "temperature": 0.1,
                    "do_sample": true,
                    "truncation": true,
                    "max_new_tokens": 200
                }
            },
            "sentence_transformer": {
                "file": "sentence-transformers/all-MiniLM-L12-v2",
                "score_threshold": 0.12
            }
        },
                {
            "name": "Science",
            "model": {
                "model_id": "facebook/bart-large-cnn",
                "task": "summarization",
                "prompt": "{text}",
                "kwargs": {
                    "temperature": 0.1,
                    "do_sample": true,
                    "truncation": true,
                    "max_new_tokens": 200
                }
            },
            "sentence_transformer": {
                "file": "sentence-transformers/all-MiniLM-L12-v2",
                "score_threshold": 0.12
            }
        }
    ]
```

For gguf models, model section should be structured as follow (selected file, over different available quantizations must be precised):

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

References can be added or deleted from a given database, adding chunks of text in the VectorDB of each associated RAG.

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
  },
  "databases": [
    {
      "name": "Litterature",
      "model": {
        "model_id": "facebook/bart-large-cnn",
        "task": "summarization",
        "prompt": "{text}",
        "kwargs": {
          "temperature": 0.1,
          "do_sample": true,
          "truncation": true,
          "max_new_tokens": 200
        }
      },
      "sentence_transformer": {
        "file": "sentence-transformers/all-MiniLM-L12-v2",
        "score_threshold": 0.12
      }
    }
  ]
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
  HuggingFacePipeline, or a gguf model) and the associated sentence transformer and created RAG (the sentence_transformer is used to create and infer vectors)
- Request:
  - Headers: 'Content-Type: multipart/form-data'
  - Body:
    `text` - text to use for prediction
    `database` - database to use for inference (model + RAG with indexed references)
- Example:
  ```
  curl 'http://0.0.0.0:5000/predict' \
  --form 'text="Who is Esmeralda?"' --form 'database="Litterature"'
  ```
- Response:
  The response contains the generated text and its associated score, and the context : details about the 5 closest vectors
  and their references used to build the generated response.
  ```json
    {
      "response": {
          "context": [
              {
                  "content": "content#1",
                  "metadata": {
                      "reference": "referenceA",
                      "score": 0.5503686535412922
                  }
              },
              {
                  "content": "content#2",
                  "metadata": {
                      "reference": "referenceB",
                      "score": 0.5342674615920695
                  }
              },
              {
                  "content": "content#3",
                  "metadata": {
                      "reference": "referenceA",
                      "score": 0.44756376562558
                  }
              },
              {
                  "content": "content#4",
                  "metadata": {
                      "reference": "referenceA",
                      "score": 0.44548622102193247
                  }
              },
              {
                  "content": "content#5",
                  "metadata": {
                      "reference": "referenceB",
                      "score": 0.3902549676845284
                  }
              }
          ],
          "score": 47,
          "text": "generated answer..."
      }
    }
  ```

You can list available databases and their uploaded references from current configuration using:
- Endpoint: `/databases`
- Method: **GET**
- Example:
  ```
  curl 'http://0.0.0.0:5000/databases'
  ```
- Reponse:
  ```
  {
      "databases": {
          "Litterature": [
              "NDame de Paris"
          ],
          "Science": []
      }
  }
  ```

###  Manage references

You can add an `.epub` document to the vector DB of the given RAG associated to a database, using:
- Endpoint: `/add_reference`
- Method: **POST**
- Request:
  - File sent on multipart
  - Body:
    `database` - database to insert reference
    `reference` - reference to insert
- Example:
  ```
  curl -F "file=@/path/data/Victor_Hugo_Notre-Dame_De_Paris_en.epub" http://0.0.0.0:5000/add_reference
  ```
- Response:
  ```
  File successfully processed
  ```

*So far, only epub files can be handled.*

You can remove a reference to the vector DB of the given RAG associated to a database, using:
- Endpoint: `/delete_reference`
- Method: **DELETE**
- Request:
  - Body:
    `database` - database to remove reference from
    `reference` - reference to delete
- Example:
  ```
  curl --form 'database="Litterature"' --form 'reference="NDame de Paris"'  http://0.0.0.0:5000/delete_reference
  ```
- Response:
  ```
  Reference successfully removed
  ```
