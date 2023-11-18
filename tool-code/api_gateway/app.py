from flask import Flask, request, Blueprint, render_template
from flask_restx import Api, Resource, fields
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
RECOMMENDATION_SERVICE_URL = "http://192.168.1.2:5004"

# Create a blueprint for the API
blueprint = Blueprint('api', __name__, url_prefix='/api')
api = Api(blueprint, version='1.0', title='Microservices Recommendation',
          description='Recommendation System for Microservices',
          doc=False)  # Disable Swagger UI

# Register the blueprint with the app
app.register_blueprint(blueprint)


query_model = api.model('Query', {
    'query': fields.String(required=True, description='Input query'),
    'encoder_choice': fields.String(default="cr_encoder", example="cr_encoder", description="Choice of cross-encoder. Options: cr_encoder, dr_encoder"),
    'model_choice': fields.String(default="BERT", example="BERT", description="Choice of sentence embedding model. Options: BERT, GPT")
})

hit_model = api.model('hit', {
    'hit': fields.String(description='Recommended hit'),
    'score': fields.Float(description='Score for the hit'),
    'correct': fields.String(description='Is the hit correct?'),
    'link': fields.String(description='GitHub link for the hit'),
    "github_name": fields.String(description='github_name'),
    "github_description": fields.String(description='github_description'),
    'github_language': fields.String(description='github_language'),
    'github_stars': fields.String(description='github_stars'),
    'readme_short': fields.String(description='Short version of README content for the hit'),
    'docker': fields.String(description='Short version of Dockerfile content for the hit'),
})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return "Test route is working!"

# Path is /api/recommend
@api.route('/recommend')
class Recommendation(Resource):
    @api.doc('recommend')
    @api.expect(query_model)
    @api.marshal_list_with(hit_model)
    def post(self):
        query = request.json['query']
        encoder_choice = request.json['encoder_choice']
        model_choice = request.json['model_choice']

        # Fetch recommendations from recommendation service
        response = requests.post(f"{RECOMMENDATION_SERVICE_URL}/recommend", json={"query": query, "encoder_choice": encoder_choice, "model_choice": model_choice})
        if response.status_code == 200:
            recommendations = response.json()
        else:
            print(f"Error: Received {response.status_code} from the service")

        return recommendations

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
