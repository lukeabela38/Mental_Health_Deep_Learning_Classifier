import os
import json
import pandas
from mentalhealth import MentalHealth
from gradio_demo import launchGUI
from flask import Flask, request, jsonify

mn = MentalHealth()
app = Flask(__name__)


@app.route("/read", methods = ["POST"])
def read():

    text = request.form.get("text")
    df = mn._runMentalHealthAlarmSystem(text)
    return dict(zip(df['Disorder'], df['Percentage Chance of Disorder']))

@app.route("/gui")
def gui():

    launchGUI()
    return "Launched GUI on http://localhost:8080"

"""
curl http://localhost:8000/read -d "text=Hello, this is a test"
"""

@app.route('/')
def hello():
    return "Mental Health Classifier using Social Media Generated Data.\n"


if __name__ == "__main__":
    from waitress import serve ## use waitress to eliminate flask production error
    serve(app, host="0.0.0.0", port = 8000)
