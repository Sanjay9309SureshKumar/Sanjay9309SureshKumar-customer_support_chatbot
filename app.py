import os
import requests
import gradio as gr
import urllib.parse
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_REPO_URL = os.getenv("GITHUB_REPO")

# Groq client
client = Groq(api_key=GROQ_API_KEY)

# Constants
FOLDER_NAMES = [
    "Accounts & Billing",
    "Design",
    "FAQs",
    "Getting Started",
    "Integrations",
    "Manage Orders",
    "Manage Shipping",
    "Products",
    "Services"
]

# Embedder
embedder = SentenceTransformer("BAAI/bge-base-en")

# Load PDFs
def load_pdfs():
    docs = []
    for folder in FOLDER_NAMES:
        print(f"Checking folder: {folder}")
        encoded_folder = urllib.parse.quote(folder)
        api_url = f"https://api.github.com/repos/Sanjay9309SureshKumar/Sanjay9309SureshKumar-customer_support_chatbot/contents/Printrove/{encoded_folder}"
        response = requests.get(api_url)
        if response.status_code == 200:
            files = response.json()
            for file in files:
                if file['name'].endswith(".pdf"):
                    print(f"Loading PDF: {file['name']}")
                    pdf_url = file['download_url']
                    pdf_text = extract_text_from_pdf(pdf_url)
                    docs.append((pdf_text, {'title': file['name'], 'url': pdf_url}))
        else:
            print(f"Failed to fetch folder: {folder} (status {response.status_code})")
    print(f"Total PDFs loaded: {len(docs)}")
    return docs

# Extract text from PDF
def extract_text_from_pdf(pdf_url):
    response = requests.get(pdf_url)
    with open("temp.pdf", "wb") as f:
        f.write(response.content)
    reader = PdfReader("temp.pdf")
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    os.remove("temp.pdf")
    return text

# Create FAISS index
def create_faiss_index(doc_tuples):
    texts = [doc[0] for doc in doc_tuples]
    metas = [doc[1] for doc in doc_tuples]
    vectors = embedder.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index, texts, metas, vectors

# Ask LLaMA 3 via Groq
def ask_llama(query, context):
    system_prompt = """You are a helpful customer support chatbot. Answer user queries based on the provided context. Be brief and friendly. Always include the most relevant PDF link below your answer."""
    full_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# Chatbot logic
def chatbot_response(query):
    docs = load_pdfs()
    index, texts, metas, vectors = create_faiss_index(docs)
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), k=1)
    matched_text = texts[I[0][0]]
    matched_meta = metas[I[0][0]]
    answer = ask_llama(query, matched_text)
    pdf_link = f"📄 [View related PDF: {matched_meta['title']}]({matched_meta['url']})"
    return answer + "\n\n" + pdf_link

# Gradio UI
demo = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(lines=2, placeholder="Ask about our services..."),
    outputs="markdown",
    title="Printrove AI Assistant",
    description="Need help? Ask about any service, and we’ll guide you with instant summaries and documents."
)

if __name__ == "__main__":
    demo.launch()
