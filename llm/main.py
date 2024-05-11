import requests

from flask import Flask, request, jsonify
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from llm.config import INDEX_DIR, EMBEDDING_MODEL, LLAMA3_URL

app = Flask(__name__)


def query_llama3(query: str) -> str:
    data = {
        'model': 'llama3',
        'prompt': query,
        'options': {'max_tokens': 200}
    }
    response = requests.post(LLAMA3_URL, json=data)
    if response.status_code == 200:
        return response.json().get('response', 'No response received.')
    else:
        return 'Error: Unable to connect to LLaMA 3 server.'


@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = FAISS.load_local(INDEX_DIR, embeddings)
    results = db.similarity_search(query)

    if results:
        return jsonify({'results': [result.page_content for result in results]})
    else:
        response = query_llama3(query)
        return jsonify({'llama3_response': response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
