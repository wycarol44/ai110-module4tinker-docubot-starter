"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import re

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def _split_into_sections(self, text):
        """
        Split a document into simple paragraph-like sections.
        We use blank lines as the separator so retrieval can target
        smaller chunks of text.
        """
        cleaned = text.strip()
        if not cleaned:
            return []

        sections = [section.strip() for section in re.split(r"\n\s*\n", cleaned) if section.strip()]
        return sections or [cleaned]

    def build_index(self, documents):
        """
        Build a tiny inverted index mapping lowercase words to the document
        sections they appear in.

        Example structure:
        {
            "token": [("AUTH.md", "section text"), ("API_REFERENCE.md", "section text")],
            "database": [("DATABASE.md", "section text")]
        }

        Keep this simple: split on blank lines, lowercase tokens,
        ignore punctuation if needed.
        """
        index = {}
        for filename, text in documents:
            for section in self._split_into_sections(text):
                tokens = set(re.findall(r"\w+", section.lower()))
                for token in tokens:
                    index.setdefault(token, []).append((filename, section))
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Return a simple relevance score for how well the text matches the query.

        Suggested baseline:
        - Convert query into lowercase words
        - Count how many appear in the text
        - Return the count as the score
        """
        if not query:
            return 0

        query_tokens = re.findall(r"\w+", query.lower())
        text_tokens = set(re.findall(r"\w+", text.lower()))

        return sum(1 for token in query_tokens if token in text_tokens)

    def retrieve(self, query, top_k=3):
        """
        Use the index and scoring function to select top_k relevant document
        sections.

        Return a list of (filename, text) sorted by score descending.
        """
        if not query:
            return []

        query_tokens = re.findall(r"\w+", query.lower())
        candidate_sections = []
        seen = set()

        for token in query_tokens:
            for filename, section in self.index.get(token, []):
                key = (filename, section)
                if key in seen:
                    continue
                seen.add(key)
                candidate_sections.append((filename, section))

        scored_results = []
        for filename, section in candidate_sections:
            score = self.score_document(query, section)
            if score > 0:
                scored_results.append((score, filename, section))

        scored_results.sort(key=lambda item: item[0], reverse=True)
        return [(filename, section) for _, filename, section in scored_results[:top_k]]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
