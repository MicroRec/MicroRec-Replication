from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder, util, models

from tqdm import tqdm
import os
from flask_cors import CORS
import torch  
import numpy as np  

app = Flask(__name__)
CORS(app)

bi_encoder_path = os.path.join("models", "bienc-exp7")
bi_encoder = SentenceTransformer(bi_encoder_path)

cr_encoder_path = os.path.join("models", "crenc-exp7")
dr_encoder_path = os.path.join("models", "crenc-readme-exp4")

cr_encoder = CrossEncoder(cr_encoder_path)
dr_encoder = CrossEncoder(dr_encoder_path)



@app.route('/encode/bi', methods=['POST'])
def bi_encode():
    query = request.json['query']
    val_text = request.json['val_text']
    emb_list = request.json['emb_list']
    val_readme = request.json['val_readme']
    encoder_choice=request.json['encoder_choice']
    model_choice = request.json['model_choice']

    if model_choice == 'GPT':
        return 
    else:
        print('BERT')
        tensor_emb_list = [torch.tensor(e, dtype=torch.float32) for e in emb_list]
        q_emb = bi_encoder.encode(query, convert_to_tensor=True)
        print("Query Embedding:", q_emb[0])
        hits = util.semantic_search([q_emb], tensor_emb_list, top_k=50+1)[0]
        print("First Hit:", hits[0])  # Print the first hit

        cross_inputs = []
        cross_readme_inputs = []
        to_remove = -1
        for hit in hits:
            readme = val_readme[hit['corpus_id']]
            text = val_text[hit['corpus_id']]
            if query == text:
                to_remove = hits.index(hit)
            cross_inputs.append([query, text])
            cross_readme_inputs.append([query, readme])

        cross_scores = cr_encoder.predict(cross_inputs)
        readme_scores = dr_encoder.predict(cross_readme_inputs)
        
        # Ensure scores are converted to float (if they are tensors or numpy arrays)
        cross_scores = [float(score) for score in cross_scores]
        readme_scores = [float(score) for score in readme_scores]
        
        for idx in range(len(cross_scores)):
            if encoder_choice=='cr_encoder':
                hits[idx]['cross_score'] = cross_scores[idx]
            else:    
                hits[idx]['readme_scores'] = readme_scores[idx]
        
        if to_remove != -1: 
            del hits[to_remove]
        hits = hits[:50]

        hit_info_list = []
        for hit in hits[:5]:
            hit_corpus_id_str = str(hit['corpus_id'])
            print(hit_corpus_id_str)
            h_text = val_text[hit['corpus_id']]
            readme = val_readme[hit['corpus_id']]
            if encoder_choice == 'cr_encoder':
                cscore_output_cr = cr_encoder.predict([[query, h_text]])
                cscore_cr = cscore_output_cr[0] if isinstance(cscore_output_cr, (list, tuple, np.ndarray)) else cscore_output_cr
                cscore=cscore_cr
            else:
                cscore_output_dr = dr_encoder.predict([[query, readme]])
                cscore_dr = cscore_output_dr[0] if isinstance(cscore_output_dr, (list, tuple, np.ndarray)) else cscore_output_dr
                cscore=cscore_dr
            
            # Find the match in df_link and extract the GitHub link if present
            #matched_link = df_link[df_link['Question Title'] == h_text]['Github Links'].iloc[0] if not df_link[df_link['Question Title'] == h_text].empty else None
            #link = matched_link[0]['link'] if matched_link and 'link' in matched_link[0] else None

            hit_info = {
                'corpus_id': hit['corpus_id'],
                'h_text_readme': readme,
                'h_text': h_text,
                'cscore': cscore,
                #'link': link
            }
            hit_info_list.append(hit_info)

        hit_info_list = sorted(hit_info_list, key=lambda x: x['cscore'], reverse=True)
        serializable_hits = []
        for hit in hit_info_list:
            serializable_hit = {k: float(v) if isinstance(v, (torch.Tensor, np.number)) else v for k, v in hit.items()}
            serializable_hits.append(serializable_hit)

        return jsonify(serializable_hits)
        # Convert hit items to serializable types
        # serializable_hits = []
        # for hit in hits:
        #     serializable_hit = {k: float(v) if isinstance(v, (torch.Tensor, np.number)) else v for k, v in hit.items()}
        #     serializable_hits.append(serializable_hit)

        # return jsonify(serializable_hits)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
