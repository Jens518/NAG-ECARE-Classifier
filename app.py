import os
from flask import Flask, request, jsonify, render_template
from classifier import ECAREClassifier

app = Flask(__name__)
classifier = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['text']
    results = classifier.classify_text(text)
    return jsonify(results)

def initialize_classifier():
    global classifier
    # Use the included Excel file in the repository
    taxonomy_file = os.path.join(os.path.dirname(__file__), "ECARE_Taxonomy_Full List.xlsx")
    classifier = ECAREClassifier(taxonomy_file)

if __name__ == "__main__":
    initialize_classifier()
    # Use port provided by Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
else:
    # For gunicorn
    initialize_classifier()
