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
[![Code Coverage](https://img.shields.io/badge/Coverage-74%25-yellow.svg)](https://github.com/alexandrainst/ragger/tree/main/tests)


Developer(s):

- Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)
- Anders Jess Pedersen (anders.j.pedersen@alexandra.dk)


## Installation

Installation with `pip`:

```bash
pip install ragger[default]@git+ssh://git@github.com/alexandrainst/ragger.git
```

Installation with `poetry`:

```bash
poetry add git+ssh://git@github.com/alexandrainst/ragger.git --extras default
```

You can replace the `default` extra with any combination of the following, to install
only the components you need:

- `postgres`
- `e5`
- `e5-cpu` (same as `e5`, but without GPU support)
- `vllm`
- `openai`
- `demo`
- `all`

For `pip`, this is done by comma-separating the extras (e.g., `ragger[vllm,demo]`),
while for `poetry`, you add multiple `--extras` flags (e.g., `--extras vllm --extras
demo`).


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


### Embedders

Embedders are used to embed documents. These can all be imported from `ragger.embedder`.

- `OpenAIEmbedder`: An embedder that uses the OpenAI Embeddings API. (default)
- `E5Embedder`: An embedder that uses an E5 model.


### Embedding Stores

Embedding stores are used to store embeddings. Embeddings are represented as objects of
the `Embedding` data class, which has an `id` and an `embedding` field. These can all be
imported from `ragger.embedding_store`.

- `NumpyEmbeddingStore`: An embedding store that stores embeddings in a NumPy array.
  (default)
- `PostgresEmbeddingStore`: An embedding store that uses a PostgreSQL database to store
  embeddings, using the `pgvector` extension. This assumes that the PostgreSQL server is
  already running, and that the `pgvector` extension is installed. See
  [here](https://github.com/pgvector/pgvector?tab=readme-ov-file#installation) for more
  information on how to install the extension.


### Generators

Generators are used to generate answers from the retrieved documents and the question.
These can all be imported from `ragger.generator`.

- `OpenAIGenerator`: A generator that uses the OpenAI Chat API. (default)
- `VllmGenerator`: A generator that uses vLLM to wrap almost any model from the Hugging
  Face Hub.


## Custom Components

You can also create custom components by subclassing the following classes:

- `DocumentStore`
- `Embedder`
- `EmbeddingStore`
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
