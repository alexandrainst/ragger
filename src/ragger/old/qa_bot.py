"""The main question-answering model, wrapping all the other modules."""

# import logging
# from collections import defaultdict
# from pathlib import Path
# from typing import Generator
#
# import pandas as pd
# from omegaconf import DictConfig
# from tqdm.auto import tqdm
#
# from .embedding import Embedder
# from .llm import answer_question_with_qa
# from .types import History
#
# logger = logging.getLogger(__name__)
#
#
# class QABot:
#     """The main question-answering model, wrapping all the other modules.
#
#     Args:
#         cfg:
#             The configuration dictionary.
#
#     Attributes:
#         cfg:
#             The configuration dictionary.
#         embedder:
#             The Embedder object.
#         qas:
#             A dictionary of questions and their answers.
#     """
#
#     def __init__(self, cfg: DictConfig) -> None:
#         """Initialises the QABot object.
#
#         Args:
#             cfg:
#                 The configuration dictionary.
#         """
#         self.cfg = cfg
#         self.embedder = Embedder(cfg=cfg)
#         self.qas: dict[str, dict[str, str]] = defaultdict(dict)
#         self._load_faqs()
#         self._embed_documents()
#         logger.info(f"QABot initialized, with {len(self.qas):,} groups.")
#
#     def __call__(self, history: History) -> str | Generator[str, None, None]:
#         """Answer a question given a context.
#
#         Args:
#             history:
#                 A list of messages in chronological order.
#
#         Returns:
#             The answer to the question.
#         """
#         question = history[-1][0]
#         assert isinstance(question, str)
#
#         # Retrieve the relevant documents from the FAQs
#         relevant_documents = self.embedder.get_relevant_documents(
#             query=question,
#             num_documents=self.cfg.poc1.embedding_model.num_documents_to_retrieve,
#         )
#
#         # If no relevant documents were found then simply return that no answer was
#         # found
#         if not relevant_documents:
#             return (
#                 "Beklager, jeg kunne ikke finde et svar på dit spørgsmål. Kan du "
#                 "omformulere det?"
#             )
#
#         return answer_question_with_qa(
#             query=question, documents=relevant_documents, cfg=self.cfg
#         )
#
#     def _embed_documents(self) -> None:
#         """Embed all the documents in the data directory, or load them from disk."""
#         groups_str = "\n\t- ".join([key.replace("\n", "") for key in self.qas.keys()])
#         logger.info(
#             f"Adding embeddings for the following {len(self.qas):,} groups:\n\t- "
#             f"{groups_str}"
#         )
#
#         # Load the embeddings from disk if they exist
#         fname = self.cfg.poc1.embedding_model.id.replace("/", "--") + ".zip"
#         embeddings_path = Path(self.cfg.dirs.data) / self.cfg.dirs.processed / fname
#         if embeddings_path.exists():
#             logger.info(f"Loading embeddings from {embeddings_path}")
#             self.embedder.load(path=embeddings_path)
#             return
#
#         for group, documents in tqdm(
#             iterable=list(self.qas.items()), desc="Embedding FAQs in all groups"
#         ):
#             questions = list(documents.keys())
#             self.embedder.add_documents(documents=questions, group=group)
#
#         self.embedder.save(path=embeddings_path)
#         logger.info(f"Embeddings saved to {embeddings_path}")
#
#     def _load_faqs(self) -> None:
#         """Load the FAQs from the Excel files in the data directory."""
#         sheet_dir = (
#             Path(self.cfg.dirs.data)
#             / self.cfg.dirs.raw
#             / self.cfg.poc1.data.excel_sheets_dir
#         )
#         for excel_file_path in tqdm(
#             iterable=list(sheet_dir.glob("*.xls*")), desc="Loading FAQs"
#         ):
#             # Skip if it is an open Excel file, i.e., if the file name starts with ~
#             if excel_file_path.name.startswith("~"):
#                 continue
#
#             excel_file = pd.ExcelFile(path_or_buffer=excel_file_path)
#             for sheet_name in tqdm(excel_file.sheet_names, leave=False):
#                 sheet = pd.read_excel(io=excel_file, sheet_name=sheet_name)
#
#                 # Check for column existence
#                 if not all([key in sheet.columns for key in ["Q", "A", "Product"]]):
#                     logger.warning(
#                         f"Skipping the sheet {sheet_name!r} in the file "
#                         f"{excel_file_path.stem!r}, as it does not contain the "
#                         "required keys."
#                     )
#                     continue
#
#                 # Check for NaN values
#                 if sheet[["Q", "A", "Product"]].isna().any().any():
#                     logger.warning(
#                         f"Skipping the sheet {sheet_name!r} in the file "
#                         f"{excel_file_path.name!r}, as it contains NaN values."
#                     )
#                     continue
#
#                 # Ensure that all values are strings
#                 sheet = sheet.astype(str)
#
#                 for _, row in sheet.iterrows():
#                     self.qas[row.Product][row.Q] = row.A
