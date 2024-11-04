# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
### Added
- Added a new abstract `Retriever` class, generalising the `Embedder` + `EmbeddingStore`
  combination, and allowing for more flexible retriever implementations. The embedder
  and embedding stores can still be used, via the `EmbeddingRetriever` retriever.
- Added new `BM25Retriever`, which uses the BM25 algorithm to retrieve documents. Note
  that this currently operates in-memory.
- Added hybrid search through the new `HybridRetriever` retriever, which can combine an
  arbitrary number of retrievers to search for documents. This currently uses the
  reciprocal rank fusion method, but it is designed to be extensible to other fusion
  methods.

### Changed
- The `RagSystem` class now only takes a `DocumentStore`, `Retriever` and `Generator`
  instance, where the `Retriever` has replaced the previous `Embedder` and
  `EmbeddingStore` combination. This allows for more flexible retriever implementations.


## [v3.0.1] - 2024-10-31
### Fixed
- Error related to `torch` import when no extras are installed.


## [v3.0.0] - 2024-10-31
### Added
- Now supports GGUF models for generators. This can be added with the new
  `GGUFGenerator` class, which works for any GGUF model on the Hugging Face Hub.
- Now uses structured generation with OpenAI models, which improves the quality of the
  generated text a tiny bit.

### Changed
- Updated the minimum bounds on the following dependencies:
	- `pydantic` from `2.8.2` to `2.9.2`
	- `tiktoken` from `0.7.0` to `0.8.0`
	- `openai` from `1.23.2` to `1.52.2`
	- `vllm` from `0.5.4` to `0.5.5`
	- `torch` from `2.3.0` to `2.4.0`
	- `psycopg2-binary` from `2.9.9` to `2.9.10`
	- `sentence-transformers` from `2.7.0` to `3.2.1`
	- `gradio` from `4.27.0` to `5.4.0`
- Removed all upper bounds from dependencies, to future-proof the package.
- Changed `poetry` dependency backend to `uv`.
- Replaced `e5` and `vllm` extras with `onprem_cpu` and `onprem_gpu` extras, for when
  you want to run the RAG system on-premises with a CPU or GPU, respectively.


## [v2.0.0] - 2024-08-21
### Added
- Added new `e5` and `cpu` extras, where `e5` installs the `sentence-transformers`
  dependency required for the `E5Embedder`, and you can add `cpu` to install the
  CPU-version of `torch` to save disk space (note that this is not available on MacOS,
  however).
- Added new `from_config` class methods to `RagSystem` and `Demo` to create instances
  from a configuration file (YAML or JSON). See the readme for more information.
- Added new `ragger-demo` and `ragger-compile` command line interfaces to run the demo
  and compile the RAG system, respectively. Compilation is useful in cases where you
  want to ensure that all components have everything downloaded and installed before
  use. Both of these take a single `--config-file` argument to specify a configuration
  file. See the readme for more information.
- Add `host` and `port` to `Demo`, which is used when the demo is launched.

### Changed
- Changed default embedder in `RagSystem` to `OpenAIEmbedder` from `E5Embedder`.

### Fixed
- Raise `ImportError` when initialising `OpenAIEmbedder` without the `openai` package
  installed.


## [v1.2.0] - 2024-08-15
### Added
- Added an `OpenAIEmbedder` that uses the OpenAI Embeddings API to embed documents.


## [v1.1.1] - 2024-08-14
### Fixed
- Fixed a bug in `NumpyEmbeddingStore` when there were fewer than `num_docs` embeddings
  in the store, causing an error when trying to retrieve embeddings.
- When calling `PostgresEmbeddingStore.clear()` or `PostgresEmbeddingStore.remove()`
  when the `embedding_dim` attribute wasn't set, it wouldn't clear/remove the store.
  This has been fixed.
- The `RagSystem.format_answer` now uses HTML `<br>` tags to separate newlines, to make
  it fully compatible to wrap in an HTML rendering context.
- `RagSystem.add_documents` now returns itself.


## [v1.1.0] - 2024-08-13
### Added
- Added a `SqliteDocumentStore` that uses a SQLite database to store documents.
- Added a `PostgresDocumentStore` that uses a PostgreSQL database to store documents.
- Added a `TxtDocumentStore` that reads documents from a single text file, separated by
  newlines.
- Added a `PostgresEmbeddingStore` that uses a PostgreSQL database to store embeddings,
  using the `pgvector` extension.

### Changed
- Added defaults to all arguments in each component's constructor, so that the
  user can create a component without specifying any arguments. This also allows for
  uniform testing of all components.


## [v1.0.0] - 2024-08-12
### Added
- Initial release, with the document store `JsonlDocumentStore`, the embedder
  `E5Embedder`, the embedding store `NumpyEmbeddingStore` and the generator
  `OpenAIGenerator`. Also features a `RagSystem` class that combines all of these
  components into a single RAG system, and a `Demo` class that provides a simple
  interface to interact with the RAG system.
