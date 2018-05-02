from flask_restful import request, Resource
from flask import *


class HeartBeat(Resource):

    def get(self):
        return render_template('index.html')
