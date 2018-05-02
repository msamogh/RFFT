from flask import Flask
from flask_restful import Api

from api.heartbeat import HeartBeat
from api.annotation import Annotation


app = Flask(__name__)
api = Api(app)

PATH_V1_API = '/api/v1'

def register_endpoints(api):
    api.add_resource(Annotation, PATH_V1_API + '/annotation/<experiment_name>/<sample_idx>')
    # api.add_resource(Train, PATH_V1_API + '/<experiment>')
    # api.add_resource()

register_endpoints(api)

app.run(host='0.0.0.0', port=8000, debug=True)
