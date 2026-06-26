import streamlit as st
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
genai.configure(api_key="abc")


def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text, chunk_size=300):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


@st.cache_resource
def create_vector_store():
    text = load_pdf("gym_data.pdf")
    chunks = chunk_text(text)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return index, chunks, model

index, chunks, embed_model = create_vector_store()


# RETRIEVE

def retrieve(query, k=10):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return "\n".join([chunks[i][:200] for i in indices[0]])

model_llm = genai.GenerativeModel("gemini-2.5-flash")

def ask_llm(prompt):
    try:
        history = ""
        for msg in st.session_state.messages[-5:]:
            history += f"{msg['role']}: {msg['content']}\n"

        full_prompt = f"""
        You are an AI Gym Coach.

        RULES:
        - Keep answers short (max 4 lines)
        - Be interactive
        - Ask questions if needed
        - Give each number of round per day in exercise
        - Explain each exercise if user ask

        Conversation:
        {history}

        {prompt}
        """

        response = model_llm.generate_content(full_prompt)
        return response.text

    except:
        return "⚠️ API Error"

def detect_intent(query):
    q = query.lower()
    if any(w in q for w in ["low", "sad", "tired", "lazy", "demotivated", "skip","motive"]):
        return "motivation"

    
    elif any(w in q for w in ["diet", "food", "meal"]):
        return "diet"

    
    elif any(w in q for w in ["exercise", "workout", "routine", "suggest","how do","how to","begin"]):
        return "exercise"

    elif "gym" in q and any(w in q for w in ["should i", "feel", "go or not"]):
        return "motivation"

    elif "gym" in q:
        return "exercise"

    return "general"

def handle_exercise(query, user_data):

    if st.session_state.get("ex_stage") is None:
        st.session_state.ex_stage = "ask_level"
        return "Are you beginner, intermediate, or advanced?"

    elif st.session_state.ex_stage == "ask_level":
        st.session_state.level = query
        st.session_state.ex_stage = "ask_days"
        return "How many days per week can you train?"

    elif st.session_state.ex_stage == "ask_days":
        days = query
        level = st.session_state.level

        st.session_state.ex_stage = None

        prompt = f"""
        User Level: {level}
        Days: {days}
        Goal: {user_data['goal']}

        Give SHORT gym plan (3-4 lines).
        """

        return ask_llm(prompt)


# DIET FLOW

def generate_diet(user_data, query):

    if st.session_state.get("diet_stage") is None:
        st.session_state.diet_stage = "ask_type"
        return "Do you prefer veg or non-veg?"

    elif st.session_state.diet_stage == "ask_type":
        diet_type = query

        st.session_state.diet_stage = None

        prompt = f"""
        Weight: {user_data['weight']} kg
        Goal: {user_data['goal']}
        Diet: {diet_type}

        Give SHORT Indian diet plan (4 lines).
        """

        return ask_llm(prompt)


def motivate_user(query, user_data):

    prompt = f"""
    User goal: {user_data['goal']}
    Situation: {query}

    Give strong motivation in 2 lines.
    """

    return ask_llm(prompt)


st.set_page_config(page_title="AI Gym Coach")
st.title("🏋️ AI Gym Chatbot")


weight = st.sidebar.number_input("Weight", 30, 150, 70)
goal = st.sidebar.selectbox("Goal", ["weight loss", "muscle gain"])

user_data = {"weight": weight, "goal": goal}

# MEMORY
if "messages" not in st.session_state:
    st.session_state.messages = []

# DISPLAY
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


query = st.chat_input("Ask anything...")

if query:
    st.chat_message("user").write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    if st.session_state.get("diet_stage") == "ask_type":
        intent = "diet"

    elif st.session_state.get("ex_stage") is not None:
        intent = "exercise"

    else:
        intent = detect_intent(query)

    with st.spinner("Thinking..."):
        if intent == "exercise":
            response = handle_exercise(query, user_data)

        elif intent == "diet":
            response = generate_diet(user_data, query)

        elif intent == "motivation":
            response = motivate_user(query, user_data)

        else:
            response = "Ask about workout, diet, or motivation 💪"

    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
