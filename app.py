import os

from flask import Flask, render_template
from flask_restful import Api

from api.annotation import Annotation
from api.train import Train
from api.experiments import Experiments


PATH_V1_API = '/api/v1'


app = Flask(__name__)
api = Api(app)

@app.route('/')
def home():
    return render_template('index.html')


def register_endpoints(api):
    api.add_resource(Experiments, PATH_V1_API + '/all_experiments')
    api.add_resource(Annotation, PATH_V1_API + '/annotation/<experiment_name>/<sample_idx>')
    api.add_resource(Train, PATH_V1_API + '/train/<experiment_name>')


register_endpoints(api)

port = int(os.environ.get('PORT', 8000))
app.run(host='0.0.0.0', port=port)
