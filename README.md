# Haystack Ollama RAG

## Setup
make dcompose

## Cleanup
make dclean

## Basic Testing
```bash
QUERY="What is the answer to 2 + 2?"
curl "http://127.0.0.1:5000/prompt" -d '{"query":"'"$QUERY"'"}'
```

## Updating RAG

Edit `rag_documents.csv`
