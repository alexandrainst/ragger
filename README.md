<a href="https://github.com/alexandrainst/ragger">
  <img
    src="https://github.com/alexandrainst/ragger/raw/main/gfx/alexandra_logo.png"
    width="239"
    height="175"
    align="right"
  />
</a>
# ragger

A repository for general-purpose RAG applications.

______________________________________________________________________
[![Code Coverage](https://img.shields.io/badge/Coverage-67%25-yellow.svg)](https://github.com/alexandrainst/ragger/tree/main/tests)


Developer(s):

- Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)
- Anders Jess Pedersen (anders.j.pedersen@alexandra.dk)


## Quick Start

Install the project as follows:

```bash
pip install git+ssh://git@github.com/alexandrainst/ragger.git
```

Then you can initialise a RAG system with default settings as follows:

```python
from ragger import RagSystem
rag_system = RagSystem()
rag_system.add_documents([
	"Copenhagen is the capital of Denmark.",
	"The population of Denmark is 5.8 million.",
	"Denmark is a member of the European Union.",
])
answer, supporting_documents = rag_system.answer("What is the capital of Denmark?")
```
