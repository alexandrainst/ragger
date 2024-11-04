<a href="https://github.com/alexandrainst/ragger">
  <img
    src="https://github.com/alexandrainst/ragger/raw/main/gfx/alexandra_logo.png"
    width="239"
    height="175"
    align="right"
  />
</a>

# Ragger

A package for general-purpose RAG applications.

______________________________________________________________________
[![Code Coverage](https://img.shields.io/badge/Coverage-73%25-yellow.svg)](https://github.com/alexandrainst/ragger/tree/main/tests)


Developer(s):

- Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)
- Anders Jess Pedersen (anders.j.pedersen@alexandra.dk)


## Installation

Installation with `pip`, `uv`, or `poetry`:

```bash
pip install ragger@git+ssh://git@github.com/alexandrainst/ragger.git
uv add git+ssh://git@github.com/alexandrainst/ragger.git
poetry add git+ssh://git@github.com/alexandrainst/ragger.git
```

You can also add additional extras to the installation, such as:

- `onprem_cpu` to use anything that requires an on-premises installation, running on a
  CPU.
- `onprem_gpu` to use anything that requires an on-premises installation, running on a
  GPU.
- `postgres` to use anything PostgreSQL-related.
- `keyword_search` to use the `BM25Retriever` for keyword-based retrieval. Note that
  this is also required when using `HybridRetriever` with the default configuration.
- `demo` to use the demo server.


### Development Installation

If you want to install the package for development, you can do so as follows:

```bash
git clone git@github.com:alexandrainst/ragger.git
cd ragger
make install
```


## Quick Start

_Checkout the `tutorial.ipynb` notebook for a more detailed guide_

Initialise a RAG system with default settings as follows:

```python
from ragger import RagSystem
rag_system = RagSystem()
rag_system.add_documents([
	"København er hovedstaden i Danmark.",
	"Danmark har 5,8 millioner indbyggere.",
	"Danmark er medlem af Den Europæiske Union."
])
answer, supporting_documents = rag_system.answer("Hvad er hovedstaden i Danmark?")
```

The `answer` is then the string answer, and the `supporting_documents` is a list of
`Document` objects that support the answer.

You can also start a demo server as follows:

```python
from ragger import Demo
demo = Demo(rag_system=rag_system)
demo.launch()
```


## Run RAG Demo With Docker

Ensure that your SSH keys are in your SSH agent, by running `ssh-add -L`. If not, you
can add them by running `ssh-add`.

You can run a CPU-based Docker container with a RAG demo with the following commands:

```bash
docker build --ssh default --build-arg config=<config-name> -t ragger -f Dockerfile.cpu .
docker run --rm -p 7860:7860 ragger
```

Here `<config-name>` is the name of a YAML or JSON file, with the following format (here
is a YAML example):

```yaml
document_store:
  name: JsonlDocumentStore
  <key>: <value>  # For any additional arguments to `JSONLDocumentStore`

retriever:
  name: EmbeddingRetriever
  embedder:
	name: OpenAIEmbedder
	<key>: <value>  # For any additional arguments to `OpenAIEmbedder`
  embedding_store:
    name: NumpyEmbeddingStore
    <key>: <value>  # For any additional arguments to `NumpyEmbeddingStore`

generator:
  name: OpenAIGenerator
  <key>: <value>  # For any additional arguments to `OpenAIGenerator`

<key>: <value>  # For any additional arguments to `RagSystem` or `Demo`
```

The config can also just be empty, to use the defaults. This is typically not
recommended, however, as you would probably need to at least specify the configuration
of your stores.

Note that if you change the configuration then you might need extra dependencies when
building your image.

Note that some components need additional environment variables to be set, such as
`OPENAI_API_KEY` for the `OpenAIEmbedder` and `OpenAIGenerator`. These can be set by
including a `.env` file in the working directory when building the Docker image, and it
will be copied into the image and used during compilation and running of the demo.

If you have any data on disk, you can simply mount it into the Docker container by
adding the `-v` flag to the `docker run` command.

To run a GPU-based Docker container, first ensure that the NVIDIA Container Toolkit is
[installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation)
and
[configured](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker).
Ensure that the the CUDA version stated at the top of the Dockerfile matches the CUDA
version installed (which you can check using `nvidia-smi`). After that, we build the
image as follows:

```bash
docker build --pull --ssh default --build-arg config=<config-name> -t ragger -f Dockerfile.cuda .
docker run --gpus 1 --rm -p 7860:7860 ragger
```


## All Available Components

Ragger supports the following components:

### Document Stores

These are the databases carrying all the documents. Documents are represented as objects
of the `Document` data class, which has an `id` and a `text` field. These can all be
imported from `ragger.document_store`.

- `JsonlDocumentStore`: A document store that reads from a JSONL file. (default)
- `SqliteDocumentStore`: A document store that uses a SQLite database to store documents.
- `PostgresDocumentStore`: A document store that uses a PostgreSQL database to store
  documents. This assumes that the PostgreSQL server is already running.
- `TxtDocumentStore`: A document store that reads documents from a single text file,
  separated by newlines.


### Retrievers

Retrievers are used to retrieve documents related to a query. These can all be imported
from `ragger.retriever`.

- `EmbeddingRetriever`: A retriever that uses embeddings to retrieve documents. These
  embeddings are computed using an embedder, which can be one of the following:
  - `OpenAIEmbedder`: An embedder that uses the OpenAI Embeddings API. (default)
  - `E5Embedder`: An embedder that uses an E5 model.

  The embeddings are stored in an embedding store, which can be one of the following:
  - `NumpyEmbeddingStore`: An embedding store that stores embeddings in a NumPy array.
	(default)
  - `PostgresEmbeddingStore`: An embedding store that uses a PostgreSQL database to
	store embeddings, using the `pgvector` extension. This assumes that the PostgreSQL
	server is already running, and that the `pgvector` extension is installed. See
	[here](https://github.com/pgvector/pgvector?tab=readme-ov-file#installation) for
	more information on how to install the extension.

- `BM25Retriever`: A retriever that uses BM25 to retrieve documents. This is keyword
  based and is thus more suitable for keyword-based queries.

- `HybridRetriever`: A retriever that fuses the results of multiple retrievers. This
  can for instance be used to combine the results of the `EmbeddingRetriever` and the
  `BM25Retriever` to get the best of both worlds (known as "Hybrid Retrieval").


### Generators

Generators are used to generate answers from the retrieved documents and the question.
These can all be imported from `ragger.generator`.

- `OpenAIGenerator`: A generator that uses the OpenAI Chat API. (default)
- `GGUFGenerator`: A generator that uses Llama.cpp to wrap any model from the Hugging
  Face Hub in GGUF format. Optimised for CPU generation.
- `VllmGenerator`: A generator that uses vLLM to wrap almost any model from the Hugging
  Face Hub. Note that this requires a GPU to run.


## Custom Components

You can also create custom components by subclassing the following classes:

- `DocumentStore`
- `Retriever` (and by extension, also `Embedder` and `EmbeddingStore`)
- `Generator`

These can then simply be added to a `RagSystem`. Here is an example:

```python
import typing
from ragger import RagSystem, DocumentStore, Document, Index

class InMemoryDocumentStore(DocumentStore):
	"""A document store that just keeps all documents in memory."""

	def __init__(self, documents: list[str]):
		self.documents = [
			Document(id=str(i), text=text) for i, text in enumerate(documents)
		]

	def add_documents(self, documents: typing.Iterable[Document]):
		self.documents.extend(documents)

	def remove(self):
		self.documents = []

	def __getitem__(self, index: Index) -> str:
		return self.documents[int(index)]

	def __contains__(self, index: Index) -> bool:
		return 0 <= int(index) < len(self.documents)

	def __iter__(self) -> typing.Generator[Document, None, None]:
		yield from self.documents

	def __len__(self) -> int:
		return len(self.documents)

document_store = InMemoryDocumentStore(documents=[
	"København er hovedstaden i Danmark.",
	"Danmark har 5,8 millioner indbyggere.",
	"Danmark er medlem af Den Europæiske Union."
])
rag_system = RagSystem(document_store=document_store)
answer, supporting_documents = rag_system.answer("Hvad er hovedstaden i Danmark?")
```
