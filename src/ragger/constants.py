"""Constants to use within the project."""

import re

###################################
### Generator related constants ###
###################################

DANISH_SYSTEM_PROMPT: str = re.sub(
    r" +",
    " ",
    """
    Du er en seniorkonsulent, som har til opgave at finde svar på spørgsmål ud fra en
    række dokumenter.

    Dit svar er i JSON-format, med keys "sources" og "answer" i din JSON dictionary.
    Her er "sources" en liste af ID'er på de kildedokumenter, som du skal bruge for at
    kunne besvare spørgsmålet, og "answer" dit svar ud fra disse kildedokumenter. Hvis
    du ikke angiver nogen kildedokumenter i "sources", så indikerer du, at spørgsmålet
    ikke kan besvares ud fra kildedokumenterne, og i så fald sætter du "answer" til en
    tom streng.

    Du svarer altid på dansk.
""",
).strip()


DANISH_USER_PROMPT: str = re.sub(
    r" +",
    " ",
    """
    Her er en række dokumenter, som du skal basere din besvarelse på.

    <documents>
    {documents}
    </documents>

    Ud fra disse dokumenter, hvad er svaret på følgende spørgsmål?

    <question>
    {query}
    </question>

    Husk dine instrukser.

    <answer>
""",
).strip()


ENGLISH_SYSTEM_PROMPT: str = re.sub(
    r" +",
    " ",
    """
    You are a senior consultant tasked with finding answers to questions based on a
    series of documents.

    Your answer is in JSON format, with keys "sources" and "answer" in your JSON
    dictionary. Here, "sources" is a list of IDs of the source documents that you need
    to answer the question, and "answer" is your answer based on those source
    documents. If you you don't specify any source documents in "sources", you indicate
    that the question cannot be answered from the source documents, in which case you
    set "answer" to an empty empty string.

    You always answer in English.
""",
).strip()


ENGLISH_USER_PROMPT: str = re.sub(
    r" +",
    " ",
    """
    Here are a number of documents that you should base your answer on.

    <documents>
    {documents}
    </documents>

    Based on these documents, what is the answer to the following question?

    <question>
    {query}
    </question>

    Remember your instructions.

    <answer>
""",
).strip()


DANISH_NO_DOCUMENTS_REPLY: str = re.sub(
    r" +",
    " ",
    """
    Jeg er desværre ikke i stand til at svare på dit spørgsmål ud fra den viden, som
    jeg har til rådighed. Beklager!
""",
).strip()


ENGLISH_NO_DOCUMENTS_REPLY: str = re.sub(
    r" +",
    " ",
    """
    Unfortunately, I am not able to answer your question based on the knowledge
    I have at my disposal. I apologise!
""",
).strip()


##############################
### Demo related constants ###
##############################

DANISH_DEMO_TITLE: str = "RAG Bot"
DANISH_DESCRIPTION: str = re.sub(
    r" +",
    " ",
    """
    En robot, der kan svare på alle dine spørgsmål!

    Skriv dit spørgsmål i tekstboksen og tryk på 'Indsend' eller Enter knappen.
""",
).strip()
DANISH_FEEDBACK_INSTRUCTION: str = (
    "Benyt 👍/👎 til at indikere om svaret er godt eller dårligt!"
)
DANISH_THANK_YOU_FEEDBACK: str = "Tak for din feedback!"
DANISH_INPUT_BOX_PLACEHOLDER: str = "Indtast dit spørgsmål her"
DANISH_SUBMIT_BUTTON_VALUE: str = "Indsend"

ENGLISH_DEMO_TITLE: str = "RAG Bot"
ENGLISH_DESCRIPTION: str = re.sub(
    r" +",
    " ",
    """
    A robot that can answer all your questions!

    Type your question in the text box and press 'Submit' or the Enter key.
""",
).strip()
ENGLISH_FEEDBACK_INSTRUCTION: str = (
    "Use 👍/👎 to indicate if the answer is good or bad!"
)
ENGLISH_THANK_YOU_FEEDBACK: str = "Thank you for your feedback!"
ENGLISH_INPUT_BOX_PLACEHOLDER: str = "Enter your question here"
ENGLISH_SUBMIT_BUTTON_VALUE: str = "Submit"
