import os

from flask import Flask, render_template
from flask_restful import Api

from jinja2.exceptions import TemplateNotFound

from api.annotation import Annotation
from api.train import Train
from api.exp import Experiment
from api.experiments import Experiments
from api.explanation import Explanation
from api.saved_experiments import SavedExperiments


PATH_V1_API = '/api/v1'


app = Flask(__name__)
api = Api(app)


@app.route('/')
def home():
    try:
        return render_template('index.html')
    except TemplateNotFound:
        return 'WHITEBox API-only server'


def register_endpoints(api):
    api.add_resource(Experiment, PATH_V1_API + '/experiment/<experiment_name>')
    api.add_resource(Experiments, PATH_V1_API + '/all_experiments')
    api.add_resource(Annotation, PATH_V1_API + '/annotation/<experiment_name>/<int:sample_idx>')
    api.add_resource(Train, PATH_V1_API + '/train/<experiment_name>')
    api.add_resource(Explanation, PATH_V1_API + '/explanation/<experiment_name>/<saved_experiment_id>')

    api.add_resource(SavedExperiments, PATH_V1_API + '/saved_experiments/<experiment_name>')


register_endpoints(api)

port = int(os.environ.get('PORT', 8000))
app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
