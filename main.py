from haystack.document_stores.in_memory import InMemoryDocumentStore
from datasets import load_dataset
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.ollama import OllamaChatGenerator

from haystack import Pipeline
import polars as pl
from flask import Flask, jsonify, request
import requests

app = Flask(__name__)

def setup_pipeline():
    document_store = InMemoryDocumentStore()
    retreived_documents = pl.read_csv("rag_documents.csv", separator=",")
    docs = []
    for row in retreived_documents.rows():
        i_id, i_title, i_content = row
        docs.append(Document(id=i_id,content=i_content,meta={"title":i_title}))
    doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    doc_embedder.warm_up()
    docs_with_embeddings = doc_embedder.run(docs)
    document_store.write_documents(docs_with_embeddings["documents"])
    #text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    retriever = InMemoryEmbeddingRetriever(document_store)

    payload = """The following are examples of wikipedia pages that are related to the query at hand.

    Context:
    {% for document in documents %}
        The page for {{ document.meta["title"] }} reads as {{ document.content }} 
    {% endfor %}

    Question: {{question}}
    Answer:
    """

    template = [ChatMessage.from_user(payload)]
    prompt_builder = ChatPromptBuilder(template=template)
    chat_generator = OllamaChatGenerator(model="qwen:1.8b", url="http://ollama:11434")
    basic_rag_pipeline = Pipeline()
    # Add components to your pipeline
    basic_rag_pipeline.add_component("text_embedder", text_embedder)
    basic_rag_pipeline.add_component("retriever", retriever)
    basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    basic_rag_pipeline.add_component("llm", chat_generator)
    # Now, connect the components to each other
    basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    basic_rag_pipeline.connect("retriever", "prompt_builder")
    basic_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")
    return basic_rag_pipeline


@app.route('/prompt', methods = ['POST'])
def propmt_rag():
    data = request.get_json(force=True)
    question = data.get("query", "")
    #question = f"{apiprompt}"
    #pipeline should be in global scope, initialized before app is run
    response = pipeline.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})
    #return response
    #result = response["llm"]["replies"][0]["_content"]["text"]
    result = response["llm"]["replies"][0].text
    #result = response["llm"]["replies"][0]
    return result

if __name__ == "__main__":
    url = "http://ollama:11434/api/pull"
    data = {
        "model": "qwen:1.8b"
    }
    response = requests.post(url, json=data)
    pipeline = setup_pipeline()
    app.run(debug = True, host="0.0.0.0")
