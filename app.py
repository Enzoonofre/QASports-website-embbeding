import streamlit as st
from datasets import load_dataset
from haystack import Pipeline
from haystack.components.readers import ExtractiveReader
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.utils import ComponentDevice  # Importando a classe ComponentDevice
import torch
from math import ceil

from utils import get_unique_docs


# Load the dataset
@st.cache_data(show_spinner=False)
def load_documents():
    """
    Load the documents from the dataset considering only unique documents.
    Returns:
    - documents: list of dictionaries with the documents.
    """
    unique_docs = set()
    dataset_name = "PedroCJardim/QASports"
    dataset_split = "basketball"
    st.caption(f'Fetching "{dataset_name}" dataset')
    # build the dataset
    dataset = load_dataset(dataset_name, dataset_split)
    # docs_validation = get_unique_docs(dataset["validation"], unique_docs)
    # docs_train = get_unique_docs(dataset["train"], unique_docs)
    docs_test = get_unique_docs(dataset["test"], unique_docs)
    half_size = ceil(len(docs_test) / 4)  # Arredondar para cima se necessário
    docs_test_half = docs_test[:half_size]  # Pegar apenas a primeira metade dos documentos
    documents = docs_test
    return documents


@st.cache_resource(show_spinner=False)
def get_document_store(documents):
    """
    Index the files in the document store.
    Args:
    - documents: list of Document objects.
    """
    # Verifique se uma GPU está disponível e use-a
    device = ComponentDevice("cuda" if torch.cuda.is_available() else "cpu")

    # Carregar o modelo no dispositivo correto (GPU ou CPU)
    st.caption(f"Building the Document Store")

    # Usando um modelo pré-treinado de Sentence Transformers, movido para a GPU (ou CPU)
    document_embedder = SentenceTransformersDocumentEmbedder(model = 'all-MiniLM-L6-v2', device=device)
    document_embedder.warm_up()

    # Criando o documento store
    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

    result = document_embedder.run(documents)

    # Acessar os documentos com os embeddings e escrever no documento store
    document_store.write_documents(result["documents"])

    return document_store


@st.cache_resource(show_spinner=False)
def get_question_pipeline(_document_store):
    """
    Create the pipeline with the retriever and reader components.
    Args:
    - doc_store: instance of the document store.
    Returns:
    - pipe: instance of the pipeline.
    """
    st.caption(f"Building the Question Answering pipeline")
    # Create the retriever and reader
    retriever = InMemoryEmbeddingRetriever(
        document_store=_document_store
    )


    # Verificar se o retriever foi criado corretamente
    print(f"Retriever: {retriever}")
    print(dir(retriever))

    reader = ExtractiveReader(model="deepset/roberta-base-squad2")
    print(f"Reader: {reader}")
    reader.warm_up()
    device = ComponentDevice("cuda" if torch.cuda.is_available() else "cpu")
    # Create the pipeline
    pipe = Pipeline()
    pipe.add_component("text_embedder", SentenceTransformersTextEmbedder(model = "all-MiniLM-L6-v2", device = device))
    pipe.add_component(instance=retriever, name="retriever")
    pipe.add_component(instance=reader, name="reader")
    pipe.connect("text_embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "reader.documents")
    return pipe


def search(pipeline, question: str):
    """
    Search for the answer to a question in the documents.
    Args:
    - pipeline: instance of the pipeline.
    - question: string with the question.
    Returns:
    - answer: dictionary with the answer.
    """
    # Get the answers
    top_k = 3
    answer = pipeline.run(
        data={
            "text_embedder": {"text": question},
            "retriever": {"top_k": 8},
            "reader": {"query": question, "top_k": top_k},
        }
    )
    max_k = min(top_k, len(answer["reader"]["answers"]))
    return answer["reader"]["answers"][0:max_k]



# Streamlit interface
_, centering_column, _ = st.columns(3)
with centering_column:
    st.image("assets/qasports-logo.png", use_column_width=True)

# Loading status
with st.status(
        "Downloading dataset...", expanded=st.session_state.get("expanded", True)
) as status:
    documents = load_documents()
    status.update(label="Indexing documents...")
    doc_store = get_document_store(documents)
    status.update(label="Creating pipeline...")
    pipe = get_question_pipeline(doc_store)
    print(pipe)
    status.update(
        label="Download and indexing complete!", state="complete", expanded=False
    )
    st.session_state["expanded"] = False

st.subheader("🔎 Basketball", divider="rainbow")
st.caption(
    """This website presents a collection of documents from the dataset named "QASports", the first large sports question answering dataset for open questions. QASports contains real data of players, teams and matches from the sports soccer, basketball and American football. It counts over 1.5 million questions and answers about 54k preprocessed, cleaned and organized documents from Wikipedia-like sources."""
)

if user_query := st.text_input(
        label="Ask a question about Basketball! 🏀",
        placeholder="How many field goals did Kobe Bryant score?",
):
    # Get the answers
    with st.spinner("Waiting"):
        try:
            answer = search(pipe, user_query)
            print(answer)
            for idx, ans in enumerate(answer):
                st.info(
                    f"""
                    Answer {idx + 1}: "{ans.data}" | Score: {ans.score:0.4f}  
                    Document: "{ans.document.meta["title"]}"  
                    URL: {ans.document.meta["url"]}
                """
                )
                with st.expander("See details", expanded=False):
                    st.write(ans)
                st.divider()
        except Exception as e:
            print(e)
