import os
import json
import pandas
from mentalhealth import MentalHealth

from flask import Flask, request, jsonify

mn = MentalHealth()
app = Flask(__name__)


@app.route("/read", methods = ["POST"])
def read():

    text = request.form.get("text")
    df = mn._runMentalHealthAlarmSystem(text)
    return dict(zip(df['Disorder'], df['Percentage Chance of Disorder']))

"""
curl http://localhost:8000/read -d "text=Hello, this is a test"
"""

@app.route('/')
def hello():
    return "Mental Health Classifier using Social Media Generated Data"


if __name__ == "__main__":
    from waitress import serve ## use waitress to eliminate flask production error
    serve(app, host="0.0.0.0", port = 8000)
