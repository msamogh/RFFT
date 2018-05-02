from flask import Flask
from flask_restful import Api

from api.annotation import Annotation
from api.train import Train


app = Flask(__name__)
api = Api(app)

PATH_V1_API = '/api/v1'

def register_endpoints(api):
    api.add_resource(Annotation, PATH_V1_API + '/annotation/<experiment_name>/<sample_idx>')
    api.add_resource(Train, PATH_V1_API + '/train/<experiment_name>')
    # api.add_resource()

register_endpoints(api)

app.run(host='0.0.0.0', port=8000, debug=True)
