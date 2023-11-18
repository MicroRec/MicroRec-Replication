
from flask import Flask, request, jsonify
import requests
from sentence_transformers import util
from flask_cors import CORS
import pandas as pd


from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder, util, models
import os
import numpy as np  


app = Flask(__name__)
CORS(app)
DATA_SERVICE_URL = "http://192.168.1.2:5002"
ENCODING_SERVICE_URL = "http://192.168.1.2:5003"




# cr_encoder_path = os.path.join("/Users/ahmedalsayed/Documents/final-code-microservice-paper/tool-code/encoding_service/models/crenc-exp7")
# dr_encoder_path = os.path.join("/Users/ahmedalsayed/Documents/final-code-microservice-paper/tool-code/encoding_service/models/crenc-readme-exp4")

# cr_encoder = CrossEncoder(cr_encoder_path)
# dr_encoder = CrossEncoder(dr_encoder_path)


def convert_to_python_floats(obj):
    if isinstance(obj, dict):
        return {k: convert_to_python_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_floats(v) for v in obj]
    elif isinstance(obj, np.float32):
        return float(obj)
    else:
        return obj






@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.json['query']
    encoder_choice = request.json['encoder_choice']
    model_choice = request.json['model_choice']
    recommendations = []

    # response = requests.post(f"{DATA_SERVICE_URL}/data/query", json={"query": query})
    # if response.status_code != 200:
    #     print(f"Error fetching data_id: {response.text}")
    #     return jsonify(recommendations)
    
    # data_id = response.json().get('data_id')

    response = requests.get(f"{DATA_SERVICE_URL}/data/val_text")
    if response.status_code != 200:
        print(f"Error fetching corpus: {response.text}")
        return jsonify(recommendations)
    val_text = response.json()

    response = requests.get(f"{DATA_SERVICE_URL}/data/val_emb_BERT")
    if response.status_code != 200:
        print(f"Error fetching corpus: {response.text}")
        return jsonify(recommendations)
    emb_list_BERT = response.json()

    response = requests.get(f"{DATA_SERVICE_URL}/data/val_emb_GPT")
    if response.status_code != 200:
        print(f"Error fetching corpus: {response.text}")
        return jsonify(recommendations)
    emb_list_GPT = response.json()

    response = requests.get(f"{DATA_SERVICE_URL}/data/val_readme")
    if response.status_code != 200:
        print(f"Error fetching corpus: {response.text}")
        return jsonify(recommendations)
    val_readme = response.json()

    if model_choice == 'BERT':
        emb_list=emb_list_BERT
    else:
        emb_list=emb_list_GPT    

    response = requests.post(f"{ENCODING_SERVICE_URL}/encode/bi", json={"query": query, "val_text": val_text,"emb_list": emb_list,"val_readme": val_readme,"encoder_choice":encoder_choice, "model_choice":model_choice})
    if response.status_code != 200:
        print(f"Error fetching bi-encoder scores: {response.text}")
        return jsonify(recommendations)
    #hits = response.json()
    hit_info_list=response.json()
    
   

    # hit_info_list = []
    # for hit in hits[:5]:
    #     hit_corpus_id_str = str(hit['corpus_id'])
    #     print(hit_corpus_id_str)
    #     h_text = val_text[hit['corpus_id']]
    #     readme = val_readme[hit['corpus_id']]
    #     if encoder_choice == 'cr_encoder':
    #         cscore_output_cr = cr_encoder.predict([[query, h_text]])
    #         cscore_cr = cscore_output_cr[0] if isinstance(cscore_output_cr, (list, tuple, np.ndarray)) else cscore_output_cr
    #         cscore=cscore_cr
    #     else:
    #         cscore_output_dr = dr_encoder.predict([[query, readme]])
    #         cscore_dr = cscore_output_dr[0] if isinstance(cscore_output_dr, (list, tuple, np.ndarray)) else cscore_output_dr
    #         cscore=cscore_dr
        
    #     # Find the match in df_link and extract the GitHub link if present
    #     #matched_link = df_link[df_link['Question Title'] == h_text]['Github Links'].iloc[0] if not df_link[df_link['Question Title'] == h_text].empty else None
    #     #link = matched_link[0]['link'] if matched_link and 'link' in matched_link[0] else None

    #     hit_info = {
    #         'corpus_id': hit['corpus_id'],
    #         'h_text_readme': readme,
    #         'h_text': h_text,
    #         'cscore': cscore,
    #         #'link': link
    #     }
    #     hit_info_list.append(hit_info)

    # hit_info_list = sorted(hit_info_list, key=lambda x: x['cscore'], reverse=True)

    # Fetch additional details from data service with error handling
    additional_details = []
    for hit in hit_info_list[:5]:
        # Use the 'h_text' field to get details
        h_text_encoded = requests.utils.quote(hit['h_text'])  # Ensure the title is URL-encoded to handle special characters
        try:
            response = requests.get(f"{DATA_SERVICE_URL}/data/details/{h_text_encoded}")
            if response.status_code == 200:
                additional_details.append(response.json())
            else:
                print(f"Failed to fetch details for hit with text '{hit['h_text']}'. Status code: {response.status_code}")
                additional_details.append({})
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            additional_details.append({})
        except ValueError as e:  # includes simplejson.decoder.JSONDecodeError
            print(f"Failed to decode JSON from the response: {e}")
            additional_details.append({})


    # Combine the hit info with the additional details
    recommendations = [
        {
            "hit": hit_info['h_text'],
            "score": hit_info['cscore'],
            "link": hit_info.get('link', None),
            "readme_short": details.get("readme_short", None),
            "docker": details.get("docker", None),
            "github_name": details.get('github_name', None),
            "github_description": details.get('github_description', None),
            "github_language": details.get("github_language", None),
            "github_stars": details.get("github_stars", None)
        }
        for hit_info, details in zip(hit_info_list[:5], additional_details)
    ]
    
    recommendations = convert_to_python_floats(recommendations)
        


    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004)
