import chromadb


client = chromadb.Client()

collection = client.get_or_create_collection(
    name="interview_docs"
)


def store_vectors(chunks):

    ids = [f"id{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        ids=ids
    )


def retrieve_docs(query):

    results = collection.query(
        query_texts=[query],
        n_results=20
    )

    docs = results["documents"][0]

    return docs