# Cosmian Confidential AI

API to run a language model in a confidential VM


## Install dependencies

By default all dependencies will be installed with the app.
```sh
pip install -r requirements.txt
```


## Build and install the app

```sh
python -m build
pip install dist/*.whl
```


## Serve the API

Be sure to have `~/.local/bin` in your `PATH`

```sh
CONFIG_PATH="./src/cosmian_ai_runner/config.json" cosmian-ai-runner
```


## Hardware optimization

When running on Intel Xeon processors, the application can leverage AMX (Advanced Matrix Extensions) to significantly
enhance performance, especially for matrix-intensive operations commonly found in machine learning workloads. To enable
this feature, simply start the application with the amx option set to true from your config file.

```sh
CONFIG_PATH="./src/cosmian_ai_runner/config.json" cosmian-ai-runner -p 5001
```

If you are running the Flask application locally using flask run, you can enable the AMX option by adding the "use_amx" key to
true from your config file.


## Config file

The config is written as a JSON file, with different parts:

### Auth

Optional part to fill the identity providers information.
If no `auth` information is present in the config file, the authentication will be disabled.

It should contain a list of the following fields per identity providers:

- `jwks_uri`: identity provider's JSON Web Key Set

- `client_id`: ID of the client calling this application set by the identity provider

### RAG documentary bases

This section enables you to define a list of Retrieval-Augmented Generation (RAG) documentary bases to be created.

In this Flask application example, we have chosen to use the Chroma document store along with the "sentence-transformers/all-MiniLM-L12-v2" model as the default sentence transformer. These choices can be modified by building your own Flask application based on this example.

However, from this flask application, you have the flexibility to configure the documentary bases you wish to set up and specify the model to be used for final inference. This allows you to tailor the application to your specific needs.

Each documentary base should include the following fields:

- `name`: display name of the documentary base

- `persist_path`: name of the file where the RAG document store will be saved locally

- `model`: name of the model used for final inference in the RAG query pipeline

- `task`: task associated with the model for final inference in the RAG query pipeline

- `kwargs`: additional keyword arguments to be passed to the model for final inference in the RAG query pipeline


Each document store are currently stored locally within a /tmp/document_store/{persist_path} folder.

### Sample config file

```json
{
  "auth": {                   // Optional to enable auth
    "openid_configs": [
      {
        "client_id": "XXXX",
        "jwks_uri": "XXXX"
      }
    ]
  },
    "use_amx": false,       // Optional - default set to false
    "hf_token": "hf_token", // Mandatory (used for translation models)
    "documentary_bases": [
    {
      "name": "Litterature",
      "persist_path": "rag_store_litterature",
      "model": "google/flan-t5-large",
      "task": "text2text-generation",
      "kwargs": {
        "max_new_tokens": 500
      }
    },
    {
      "name": "Science",
      "persist_path": "rag_store_science",
      "model": "google/flan-t5-large",
      "task": "text2text-generation",
      "kwargs": {
        "max_new_tokens": 500
      }
    }
  ]
}
```

## API Endpoints

### Summarize text

Summary is made using "facebook/bart-large-cnn" model.

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

Translation is made using one of the "Helsinki-NLP/opus-mt" models.

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
  --form 'query="Who is Elara?"' --form 'context="Elara is a girl living in a forest..."'
  ```
- Response:
  The response contains the answer to the query, from given context.
  ```json
    {
      "result": ["Elara is the sovereign of the mystical forests of Eldoria"]
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
  --form 'query="Who is Esmeralda?"' --form 'db="Litterature"'
  ```
- Response:
  The response contains the answer to the query, from given context.
  ```json
    {
      "result": ["a street dancer"]
    }
  ```

You can list available documentary basis and their uploaded references from current configuration using:
- Endpoint: `/documentary_basis`
- Method: **GET**
- Example:
  ```
  curl 'http://0.0.0.0:5000/documentary_bases'
  ```
- Reponse:
  ```
  {
      "documentary_bases": {
          "Litterature": [
              "NDame de Paris"
          ],
          "Science": []
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
  curl -F "file=@/data/doc.pdf" --form 'db="Litterature"' --form 'reference="crypto"'  http://0.0.0.0:5000/add_reference
  ```
- Response:
  ```
  File successfully processed
  ```

*So far, only epub, pdf and docx files can be handled.*

You can remove a reference to the vector DB of the given RAG associated to a database, using:
- Endpoint: `/delete_reference`
- Method: **DELETE**
- Request:
  - Body:
    `db` - database to remove reference from
    `reference` - reference to delete
- Example:
  ```
  curl --form 'db="Litterature"' --form 'reference="NDame de Paris"'  http://0.0.0.0:5000/delete_reference
  ```
- Response:
  ```
  Reference successfully removed
  ```
