# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


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
