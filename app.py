import os

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

from ai import load_ai, run_ai

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(APP_ROOT, "dist"), static_folder=os.path.join(APP_ROOT, "dist"))
app.config.from_pyfile('config.py')

CORS(app)

enc, nsamples, batch_size, hparams, temperature, top_k, model_name = load_ai()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate():
    """
    enc = app.config['enc']
    nsamples = app.config['nsamples']
    batch_size = app.config['batch_size']
    hparams = app.config['hparams']
    temperature = app.config['temperature']
    top_k = app.config['top_k']
    model_name = app.config['model_name']
    """
    data = request.get_json(force=True)
    text = data.get('text')
    length = data.get('length')
    if not length:
        length = 250
    output_text = run_ai(enc, nsamples, batch_size, length, hparams, temperature, top_k, model_name, text)
    return jsonify({'generated': output_text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
