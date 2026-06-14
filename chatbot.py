import os
import streamlit as st
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


genai.configure(api_key="ABC")
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def split_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

@st.cache_resource
def create_vector_store(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, model

#  Retrieve relevant chunks
def retrieve(query, index, model, chunks, k=10):
    q_embed = model.encode([query])
    D, I = index.search(np.array(q_embed), k)
    return [chunks[i] for i in I[0]]


def generate_answer(query, context):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
    You are an intelligent E-commerce assistant.

    Instructions:
    - Use ONLY the given context
    - Understand the user's intent (price, category, etc.)
    - If user asks for "under X", find products with price less than X
    If user asks "is it good?", check ratings
    - Rating > 4.0 → Good product
    - Rating 3.0–4.0 → Average
    - Rating < 3.0 → Not good
    - Always mention ratings in your answer
    - If user ask for"payment" ,then show payment option of just above product
    - Extract product names and prices from context
    - Give all best matching products
    - If user ask-Tasks:
    1. Recommend best products based on query
    2. If user says "best", prioritize:
    - Higher ratings
    - Better features
    3. If price condition exists, follow it
    Rules:
    - Use ONLY context
    - Extract product name, price, ratings
    -Show 2–3 best options
    For comparison:
    - Show clear differences (price, rating, features)


    If no relevant data exists, say:
    "I am sorry, that information is not in my database."

    Context:
    {context}

    Question:
    {query}
    """

    response = model.generate_content(prompt)
    return response.text


def main():
    st.set_page_config(page_title="E-COMMERCE ADVISOR CHATBOT", layout="wide")
    st.title("E-COMMERCE ADVISOR CHATBOT")

    #  Upload PDF
    uploaded_file = "chatbotdata.pdf"

    if uploaded_file:
        text = load_pdf(uploaded_file)
        chunks = split_text(text)

        index, embed_model = create_vector_store(chunks)

        # Chat memory
        if "history" not in st.session_state:
            st.session_state.history = []
            st.session_state.history.append(("Bot", "Hi 👋, Confused about what to buy?  Don't worry I am here to help you,Please type the Question below"))
        if "last_product" not in st.session_state:
            st.session_state.last_product = ""    

        query = st.chat_input("Ask your question")
        if query:
            if "above" in query.lower() or "it" in query.lower():
                if st.session_state.last_product:
                    query = f"{query} about {st.session_state.last_product}"

        if query:
            context_list = retrieve(query, index, embed_model, chunks)
            context = "\n\n".join(context_list)

            answer = generate_answer(query, context)
            #  Detect product name from answer
            products = ["Nothing Phone 1","iQOO Neo 7","Samsung Galaxy S23","iPhone 13","OnePlus 11R"]
            for p in products:
                if p.lower() in answer.lower():
                    st.session_state.last_product = p

            st.session_state.history.append(("You", query))
            st.session_state.history.append(("Bot", answer))

        
        for role, msg in st.session_state.history:
            if role == "You":
                with st.chat_message("user"):
                    st.markdown(f"**You:** {msg}")
            else:
                with st.chat_message("assistant"):
                    st.markdown(f"**Bot:** {msg}")

        
        with st.sidebar:
            if st.button("🧹 Clear Chat"):
                st.session_state.history = []

if __name__ == "__main__":
    main()