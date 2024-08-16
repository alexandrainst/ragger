# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
### Added
- Added new `e5` and `e5-cpu` extras to install the `E5Embedder` with and without GPU
  support, respectively.
- Added new `from_config` class methods to `RagSystem` and `Demo` to create instances
  from a configuration file (YAML or JSON). See the readme for more information.
- Added new `ragger-demo` and `ragger-compile` command line interfaces to run the demo
  and compile the RAG system, respectively. Compilation is useful in cases where you
  want to ensure that all components have everything downloaded and installed before
  use. Both of these take a single `--config-file` argument to specify a configuration
  file. See the readme for more information.
- Added new `e5` and `cpu` extras, where `e5` installs the `sentence-transformers`
  dependency required for the `E5Embedder`, and you can add `cpu` to install the
  CPU-version of `torch` to save disk space.

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
