# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [v1.0.0] - 2024-08-12
### Added
- Initial release, with the document store `JsonlDocumentStore`, the embedder
  `E5Embedder`, the embedding store `NumpyEmbeddingStore` and the generator
  `OpenAIGenerator`. Also features a `RagSystem` class that combines all of these
  components into a single RAG system, and a `Demo` class that provides a simple
  interface to interact with the RAG system.
