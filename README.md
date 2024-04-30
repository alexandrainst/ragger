<a href="https://github.com/alexandrainst/ragger"><img src="https://github.com/alexandrainst/ragger/raw/main/gfx/alexandra_logo.png" width="239" height="175" align="right" /></a>
# ragger

A repository for general-purpose RAG applications.

______________________________________________________________________
[![Code Coverage](https://img.shields.io/badge/Coverage-79%25-yellowgreen.svg)](https://github.com/alexandrainst/ragger/tree/main/tests)


Developer(s):

- Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)
- Anders Jess Pedersen (anders.j.pedersen@alexandra.dk)


## Quick Start

1. Install the project using the `make install` command.
2. Activate the newly created Python virtual environment with `source
   .venv/bin/activate`.
3. Run the demo using the command `python src/scripts/run_demo.py`.
4. A link will appear which takes you to the demo.


## Docker

The RAG demo can also be run using Docker, using the following steps:

1. Build the Docker image using `docker build --tag rag-demo`.
2. Run the demo using `docker run --name rag-demo`


## Overview

The general structure of the repository is based on the following classes:

- The `RagSystem` class in the `rag_system` module is the main entry point for the RAG
  system, and orchestrates all the other parts.
- The `DocumentStore` class in the `document_store` module is a database with documents.
- The `Embedder` class in the `embedder` module converts documents to embeddings.
- The `EmbeddingStore` class in the `embedding_store` module is a database with
  embeddings.
- The `Generator` class in the `generator` module generates answers from a query and a
  list of relevant documents.
- The `Demo` class in the `demo` module wraps the `RagSystem` in a Gradio demo that can
  be accessed in the browser. The demo has three modes (`demo.mode`), `no_feedback`, `feedback` and `strict_feedback`. `strict_feedback` requires the use to give feedback before the are allowed to ask further questions, `feedback` makes feedback optional, and `no_feedback` removes the feedback option.

The `DocumentStore`, `Embedder`, `EmbeddingStore` and `Generator` classes are all
abstract, and can be subclassed to a concrete implementation, which depends on the
concrete use case. Here are the currently supported concrete classes:

- `DocumentStore`:
    1. `JsonlDocumentStore`, which assumes the document store is a single JSONL file on
       disk, which is loaded into memory.
- `Embedder`:
    1. `E5Embedder`, which embeds the documents using an E5 sentence transformer model.
- `EmbeddingStore`:
    1. `NumpyEmbeddingStore`, which stores the embeddings in a Numpy array in memory.
- `Generator`:
    1. `OpenAIGenerator`, which generates embeddings using an OpenAI model.

The concrete classes can be found in the same modules as the abstract ones.

The configuration of all of these components can be found in the `config/config.yaml`
file. This is a [Hydra](https://hydra.cc) configuration, meaning that you can adjust
the hyperparameters directly when running the demo script. For instance, to adjust the
temperature of the OpenAI model to 1, you can run the demo as

```
python src/scripts/run_demo.py generator.openai.temperature=1.0
```

You can easily switch between different document stores, embedders, embedding stores or
generators, using the `type` variable in the configuration. For instance, to change
from the OpenAI generator to an (at the moment non-existing) Open Source generator, we
can run the script as follows:

```
python src/scripts/run_demo.py generator.type=open_source
```
