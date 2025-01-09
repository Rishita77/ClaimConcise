import os
from PyPDF2 import PdfReader
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

def get_openai_key():
    return "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"

def summarize_document(file_path):
    open_api_key = get_openai_key()
    os.environ["OPENAI_API_KEY"] = open_api_key

    pdfreader = PdfReader(file_path)
    text = ""

    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            text += content

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
    chunks = text_splitter.create_documents([text])

    chunks_text = [doc.page_content for doc in chunks]
    combined_text = " ".join(chunks_text)

    single_prompt = '''
    Please summarize the following insurance policy document, focusing on these key points:
    1. Determine the amount of coverage needed based on fixed expenses, debts, and future financial goals.
    2. Whether the coverage amount accounts for inflation to maintain financial security over time.
    3. Insurance details.
    4. Claim settlement ratio.
    5. Additional features or riders that may enhance coverage, such as critical illness benefits or premium waivers.
    6. Terms and Conditions: Carefully read the fine print of the policy to understand exclusions, limitations, and specific terms that could affect coverage during a claim.
    7. Exclusions.
    8. Documentation Requirements during claims.

    Policy: `{text}`
    Summary:
    '''

    output_new = llm(single_prompt.format(text=combined_text))
    return output_new.content
