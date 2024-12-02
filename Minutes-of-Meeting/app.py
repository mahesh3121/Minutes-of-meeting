from flask import Flask, render_template, jsonify
import subprocess
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run-script', methods=['GET'])
def run_script():
    try:
        # Run the `parth.py` script to open the Tkinter GUI
        subprocess.Popen(["python", "parth.py"], shell=True)  # Use shell=True for Windows compatibility
        return jsonify({"status": "success", "message": "Please wait till the TKinter GUI opens!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
