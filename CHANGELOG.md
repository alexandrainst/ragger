# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
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
