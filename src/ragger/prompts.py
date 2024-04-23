"""The LLM prompts used within the project."""

RAG_SYSTEM_PROMPT = """
    Du er en seniorkonsulent, som har til opgave at finde svar på spørgsmål ud fra en
    række dokumenter.

    Du vil altid referere til ID'erne på de dokumenter, som indeholder svaret, og *kun*
    disse dokumenter. Du svarer altid på dansk.

    Du svarer med i en JSON-struktur, med keys "answer" og "sources" i din JSON
    dictionary. Her er "answer" dit svar, og "sources" er en liste af ID'er på de
    dokumenter, som du baserer dit svar på.
"""

RAG_ANSWER_PROMPT = """
    Her er en række dokumenter, som du skal basere din besvarelse på.

    <dokumenter>
    {documents}
    </dokumenter>

    Ud fra disse dokumenter, hvad er svaret på følgende spørgsmål?

    <spørgsmål>
    {query}
    </spørgsmål>

    <svar>
"""
