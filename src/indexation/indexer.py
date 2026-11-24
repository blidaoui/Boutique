
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Configuration
BASE_DIR = Path(__file__).resolve().parents[2]
FICHIER_PDF = os.getenv("FICHIER_PDF", str(BASE_DIR / "doc" / "Politique_boutique.pdf"))
EMBEDDINGS = os.getenv("EMBEDDINGS", "gemini-embedding-001")
CHROMA_DIR = os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db"))
COLLECTION = os.getenv("COLLECTION", "politique_boutique")


# 1. Connexion à ChromaDB
def connecter_chroma():

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDINGS
    )

    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


# 2. Chargement et découpage du PDF
def charger_et_decouper():

    if not os.path.exists(FICHIER_PDF):
        raise FileNotFoundError(
            f"PDF introuvable : {FICHIER_PDF}"
        )

    docs = PyPDFLoader(FICHIER_PDF).load()

    print("Pages chargées :", len(docs))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    print("Chunks :", len(chunks))

    return chunks


# 3. Création de l'index
def creer_index():

    chunks = charger_et_decouper()

    db = connecter_chroma()

    db.reset_collection()

    db.add_documents(chunks)

    print(
        f"Index ChromaDB créé : {len(chunks)} chunks "
        f"dans '{COLLECTION}' ({CHROMA_DIR})"
    )

    return db


# 4. Test rapide
if __name__ == "__main__":

    db = creer_index()

    question = "Puis-je retourner un produit après 20 jours ?"

    print(f"\nTest : {question}")

    results = db.similarity_search(question, k=2)

    for document in results:
        print(
            " •",
            document.page_content.replace("\n", " ")[:120],
            "..."
        )