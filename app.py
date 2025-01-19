from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from summarization import summarize_document 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

app = Flask(__name__)

# Ensure the uploads folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

load_dotenv()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_openai_key():
    return os.getenv("OPENAI_API_KEY")

# Set OpenAI API key globally
open_api_key = get_openai_key()
os.environ["OPENAI_API_KEY"] = open_api_key

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Check if the upload folder is writable
        if not os.access(app.config['UPLOAD_FOLDER'], os.W_OK):
            return jsonify({"error": "Cannot write to upload folder"}), 500

        try:
            file.save(file_path)
        except Exception as e:
            return jsonify({"error": f"Error saving file: {str(e)}"}), 500

        # Summarize the document
        summary = summarize_document(file_path)

        return jsonify({"summary": summary})

    return jsonify({"error": "Invalid file format"}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    # Retrieve data from JSON body
    data = request.get_json()
    
    question = data.get('question')
    document_summary = data.get('document_summary')
    
    if not question or not document_summary:
        return jsonify({"error": "Missing question or document summary"}), 400

    follow_up_prompt = f"""
    The following is a summary of an insurance policy document:
    {document_summary}
    
    Based on this summary, answer the following question:
    Question: {question}
    Answer:
    """

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    response = llm(follow_up_prompt)  # Pass the context directly to the model

    return jsonify({"response": response.content})

if __name__ == '__main__':
    app.run(debug=True)
