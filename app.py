from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from summarization import summarize_document 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Summarize the document
        summary = summarize_document(file_path)

        return jsonify({"summary": summary})

    return jsonify({"error": "Invalid file format"})

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form['question']
    document_summary = request.form['document_summary']
    

    follow_up_prompt = f"""
    The following is a summary of an insurance policy document:
    {document_summary}
    
    Based on this summary, answer the following question:
    Question: {question}
    Answer:
    """
    open_api_key = get_openai_key()
    os.environ["OPENAI_API_KEY"] = open_api_key
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    response = llm(follow_up_prompt)  # Pass the context directly to the model

    return jsonify({"response": response.content})

load_dotenv()

def get_openai_key():
    return os.getenv("OPENAI_API_KEY")

if __name__ == '__main__':
    app.run(debug=True)
