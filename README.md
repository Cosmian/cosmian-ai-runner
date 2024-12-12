# Cosmian AI runner

Confidential computing backend to run AI models

## Structure

The AI Runner is a Flask-based application that provides endpoints for performing inference across various AI tasks and pipelines, such as text summarization, translation, text querying, and retrieval-augmented generation (RAG) over document databases.

These pipelines are constructed using the Haystack library (https://haystack.deepset.ai/) as a foundation. Users have the flexibility to customize the pipelines by modifying the loaded models, selecting a preferred vector database, and tailoring the setup to meet specific requirements.

## Usage

- Build and install the [app](./app/README.md)

- Edit the config file ([more info](./app/README.md#config-file))

- Run the app

```bash
CONFIG_PATH="./run/config.json" cosmian-ai-runner --port 5001
Using current model, you need to add your HuggingFace token as an env variable (HF_API_TOKEN). 
```

Details of the API Endpoints are explained in the `app/` folder of the repository.
