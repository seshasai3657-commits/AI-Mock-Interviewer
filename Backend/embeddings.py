from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


model = SentenceTransformer("all-MiniLM-L6-v2")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


def create_chunks(text):

    chunks = splitter.split_text(text)

    return chunks


def create_embeddings(chunks):

    embeddings = model.encode(chunks)

    return embeddings