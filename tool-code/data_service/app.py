from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification,AutoTokenizer,AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from tqdm import tqdm
import string
from sklearn.feature_extraction import _stop_words as stop_words
from rank_bm25 import BM25Okapi
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import requests
from flask import Flask, request, jsonify
import os
import pandas as pd
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

bi_encoder_path = os.path.join("/UsersDocuments/final-code-microservice-paper/tool-code/encoding_service/models/bienc-exp7")
bi_encoder = SentenceTransformer(bi_encoder_path)


data_folder = 'data/'


df = pd.read_excel(os.path.join(data_folder, "20231004_data.xlsx"), index_col=0)
df['readme_short'] = df['readme_short'].astype(str)
df['docker'] = df['docker'].astype(str)

df['answer'] = df.apply(lambda row: ' '.join(row['readme_short'].split()[:300] + row['docker'].split()[:200]), axis=1)

english_stopwords = set(stopwords.words('english'))
def bm25_tokenizer(text):
  tokenized_doc = []
  for token in text.lower().split():
    token = token.strip(string.punctuation)

    if len(token) > 0 and token not in english_stopwords:
      tokenized_doc.append(token)
      
  return tokenized_doc


with open(os.path.join(data_folder, 'test_passage.json'), 'r') as f:
    val_passage = json.load(f)
with open(os.path.join(data_folder, 'test_corpus.json'), 'r') as f:
    val_corpus = json.load(f)


# Construct val_query_answer dictionary

val_query_answer = {}
val_query_readme = {}
for idx, rel in val_passage.items():
    pos = rel[0]

    readme = df.loc[int(pos[0]), 'answer']
    questions = []
    for p in pos:
        question = df.loc[int(p), 'Question Title']
        questions.append(question)
    val_query_answer[idx] = questions
    val_query_readme[idx] = [readme]


val_text = list(val_corpus.values())

val_readme = []
for t in val_text:
    val_readme.append(df.loc[df['Question Title'] == t, 'answer'].values[0])

    to_emb = val_text
val_emb = bi_encoder.encode(to_emb, show_progress_bar=True, convert_to_tensor=True)

from tqdm import tqdm
tokenized_corpus = []
for idx, passage in tqdm(val_corpus.items()):
    tokenized_corpus.append(bm25_tokenizer(passage))




@app.route('/data/val_query_answer', methods=['GET'])
def get_val_query_answer():
    return jsonify(val_query_answer)

@app.route('/data/query', methods=['POST'])
def get_data_id():
    query = request.json['query']
    matching_rows = df[df['Question Title'] == query]
    if not matching_rows.empty:
        data_id = str(matching_rows.index.values[0])
    else:
        data_id = None
    return jsonify({"data_id": data_id})

@app.route('/data/val_text', methods=['GET'])
def get_val_text():
    return jsonify(val_text)

@app.route('/data/val_emb_BERT', methods=['GET'])
def get_val_emb_BERT():
    emb_list_BERT = val_emb.detach().numpy().tolist()
    return jsonify(emb_list_BERT)

@app.route('/data/val_emb_GPT', methods=['GET'])
def get_val_emb_GPT():
    emb_list_GPT = val_emb.detach().numpy().tolist()
    return jsonify(emb_list_GPT)

@app.route('/data/val_readme', methods=['GET'])
def get_val_readme():
    return jsonify(val_readme)



def get_github_details(link):
    # Extract username and repo name from the link
    parts = link.split('/')
    if len(parts) < 2:
        return None  # Not a valid GitHub repo URL
    username, repo_name = parts[-2], parts[-1]

    # Make a request to the GitHub API
    url = f"https://api.github.com/repos/{username}/{repo_name}"
    headers = {'Accept': 'application/vnd.github.v3+json'}
    response = requests.get(url, headers=headers)

    # Check the rate limit
    remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
    if remaining == 0:
        print("Hit GitHub API rate limit!")
        return None

    # If the request was successful, extract the language and stars
    if response.status_code == 200:
        data = response.json()
        return {
            'name': data.get('name'),
            'description': data.get('description'),
            'language': data.get('language'),
            'stars': data.get('stargazers_count')
        }
    else:
        print(f"Failed to fetch details for {link}. Status code: {response.status_code}")
        return None

@app.route('/data/details/<path:encoded_title>', methods=['GET'])
def get_details(encoded_title):
    title = requests.utils.unquote(encoded_title) 
    # Load Results_Tags_with_Docker_S7_final_data.json and find the data by id
    with open(os.path.join('/UsersDocuments/final-code-microservice-paper/experiments/data/Results_Tags_with_Docker_S7_final_data.json'), 'r') as f:
        results_data = json.load(f)

        link_data = next((item for item in results_data if item["Question Title"] == title), None)

    if not link_data:
        return jsonify({"error": "Data not found"}), 404

    # Extract the details from the link_data
    github_link = link_data["Github Links"][0]["link"] if link_data["Github Links"] else None
    readme = link_data["Github Links"][0]["readme_content"] if link_data["Github Links"] else None
    dockerfile = link_data["Github Links"][0]["docker_related"] if link_data["Github Links"] else None

    # Truncate the readme and dockerfile content to the first 50 words
    def truncate_text(text):
        return ' '.join(text.split()[:50]) + '...' if text and len(text.split()) > 50 else text
    
    readme_short = truncate_text(readme)
    dockerfile_short = truncate_text(dockerfile)

    # Get additional details from GitHub
    github_details = get_github_details(github_link) if github_link else {}

    return jsonify({
        "link": github_link,
        "readme_short": readme_short,
        "docker": dockerfile_short,
        "github_name": github_details.get('name') if github_details else None,
        "github_description": github_details.get('description') if github_details else None,
        "github_language": github_details.get('language') if github_details else None,
        "github_stars": github_details.get('stars') if github_details else None
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)