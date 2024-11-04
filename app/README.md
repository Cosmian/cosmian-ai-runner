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

### Predict using text as context

- Endpoint: `/context_predict`
- Method: **POST**
- Description: get prediction from a model using current text as a context
- Request:
  - Headers: 'Content-Type: multipart/form-data'
  - Body:
    `context` - text to use as context for prediction
    `query` - query to answer
- Example:
  ```
  curl 'http://0.0.0.0:5000/context_predict' \
  --form 'text="Who is Elara?"' --form 'context="Elara is a girl living in a forest..."'
  ```
- Response:
  The response contains the answer to the query, from given context.
  ```json
    {
      "response": ["Elara is the sovereign of the mystical forests of Eldoria"]
    }
  ```

### Predict using RAG

- Endpoint: `/rag_predict`
- Method: **POST**
- Description: get prediction from a model using RAG and configured documentary basis
- Request:
  - Headers: 'Content-Type: multipart/form-data'
  - Body:
    `db` - documentary basis to use for prediction
    `query` - query to answer
- Example:
  ```
  curl 'http://0.0.0.0:5000/rag_predict' \
  --form 'text="Who is Esmeralda?"' --form 'db="litterature"'
  ```
- Response:
  The response contains the answer to the query, from given context.
  ```json
    {
      "response": ["a street dancer"]
    }
  ```

You can list available documentary basis and their uploaded references from current configuration using:
- Endpoint: `/documentary_basis`
- Method: **GET**
- Example:
  ```
  curl 'http://0.0.0.0:5000/documentary_basis'
  ```
- Reponse:
  ```
  {
      "documentary_basis": {
          "litterature": [
              "NDame de Paris"
          ],
          "science": []
      }
  }
  ```

###  Manage references

You can add an `.epub` document, `.docx` document or a PDF to the vector DB of the given RAG associated to a database, using:
- Endpoint: `/add_reference`
- Method: **POST**
- Request:
  - File sent on multipart
  - Body:
    `db` - database to insert reference
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
    `db` - database to remove reference from
    `reference` - reference to delete
- Example:
  ```
  curl --form 'database="Litterature"' --form 'reference="NDame de Paris"'  http://0.0.0.0:5000/delete_reference
  ```
- Response:
  ```
  Reference successfully removed
  ```
