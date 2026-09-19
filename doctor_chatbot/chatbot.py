"""Doctor Chatbot - RAG based medical assistant.

Pipeline:
  data/ docs (.pdf, .docx, .txt, .md)
    -> load -> split -> Google embeddings -> FAISS -> retriever -> Gemini LLM

Run:
  .\\env\\Scripts\\python doctor_chatbot\\chatbot.py
  Type 'exit' to quit. Type 'reload' to re-ingest data/ folder.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root and from this folder (whichever exists)
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
load_dotenv(ROOT_DIR / ".env")
load_dotenv(BASE_DIR / ".env")

# langchain-google-genai expects GOOGLE_API_KEY, repo uses GEMINI_API_KEY.
# Accept either one.
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

SYSTEM_PROMPT = """You are a helpful Doctor Chatbot assistant.
Use ONLY the provided medical context to answer the user's question.
If the answer is not in the context, say you don't know and advise
consulting a qualified doctor. Never give a definitive diagnosis;
always add a short safety note to consult a healthcare professional
for serious or persistent symptoms.

Context:
{context}

Question: {question}

Answer in simple, caring language. Include a disclaimer when the
question is about diagnosis or treatment.
"""

prompt = PromptTemplate.from_template(SYSTEM_PROMPT)


def load_documents(data_dir: Path):
    """Load .pdf, .docx, .txt, .md files from data_dir."""
    docs = []
    if not data_dir.exists():
        return docs

    for f in sorted(data_dir.iterdir()):
        if f.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(f)).load())
        elif f.suffix.lower() == ".docx":
            try:
                docs.extend(Docx2txtLoader(str(f)).load())
            except Exception as e:
                print(f"Skipped {f.name}: {e}")
        elif f.suffix.lower() in (".txt", ".md"):
            docs.extend(TextLoader(str(f), encoding="utf-8").load())
    return docs


def build_vectorstore(docs):
    """Split docs, embed with Gemini, build FAISS index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("No text found in data/ folder. Add .pdf/.docx/.txt files first.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(VECTORSTORE_DIR))
    return db


def get_vectorstore():
    """Load existing FAISS index or build a new one from data/."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    if VECTORSTORE_DIR.exists() and (VECTORSTORE_DIR / "index.faiss").exists():
        return FAISS.load_local(
            str(VECTORSTORE_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    print("No vectorstore found. Ingesting data/ folder...")
    docs = load_documents(DATA_DIR)
    if not docs:
        raise FileNotFoundError(
            f"No documents in {DATA_DIR}. Add medical .pdf/.txt/.docx files first."
        )
    return build_vectorstore(docs)


def answer_question(llm, retriever, question: str) -> str:
    """Retrieve context and ask the LLM."""
    retrieved = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in retrieved)
    response = llm.invoke(prompt.format(context=context, question=question))

    if isinstance(response.content, list):
        return "".join(
            block["text"] for block in response.content if block.get("type") == "text"
        )
    return response.content


def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: Set GOOGLE_API_KEY (or GEMINI_API_KEY) in .env first.")
        return

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    db = get_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 4})

    print("Doctor Chatbot ready (RAG). Type 'exit' to quit, 'reload' to re-ingest.")
    while True:
        user_input = input("\nPatient: ").strip()
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "reload":
            docs = load_documents(DATA_DIR)
            db = build_vectorstore(docs)
            retriever = db.as_retriever(search_kwargs={"k": 4})
            print("Knowledge base reloaded.")
            continue
        if not user_input:
            continue
        try:
            print("\nDoctor:", answer_question(llm, retriever, user_input))
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
