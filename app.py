import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader

# Initialize Google Gemini model
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Sidebar for API Key input and tab selection
with st.sidebar:
    tabs = st.selectbox("Choose a tab", ["Main Page", "File Q&A", "Chatbot", "Practice Exam Generator", "Video Explanation Generator", "Text Simplifier"])
    api_key = st.text_input("Google API Key", key="gemnikey", type="password")

# Main Page Tab
if tabs == "Main Page":
    st.title("🌬️ Ease Platform for Speacial Students ")
    st.write("""
        Welcome to the Ease Platform! 
        
        - **File Q&A**: Upload an article and get answers to your questions in a simplified manner.
        - **Chatbot**: Interact with a chatbot for personalized assistance and inquiries.
        - **Practice Exam Generator**: Generate practice exams based on difficulty, subject, and topic to aid learning.
        - **Video Explanation Generator**: Generate audio explanations with text-to-speech functionality for better understanding.
        - **Text Simplifier**: Simplify text to make it more accessible and easier to understand.

        Select a tab from the sidebar to get started!
    """)

# File Q&A Tab
elif tabs == "File Q&A":
    st.title("📝 File Q&A")
    uploaded_file = st.file_uploader("Upload an article", type=("txt", "md", "pdf"))
    question = st.text_input("Ask something about the article", placeholder="Can you give me a short summary?", disabled=not uploaded_file)
    
    if uploaded_file and question and api_key:
        article = None
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            article = "".join([page.extract_text() for page in pdf_reader.pages])
        else:
            article = uploaded_file.read().decode('utf-8')
        
        if article:
            genai.configure(api_key=api_key)
            prompt_text = f"SYSTEM: Summarize the following article in simple terms for students with Global Developmental Delay. Use simpler vocabulary.\nUSER: {question}\nARTICLE: {article}"
            response = model.generate_content(prompt_text).text
            st.write("### Answer")
            st.write(response)
        else:
            st.error("Couldn't extract article.")

# Chatbot Tab
elif tabs == "Chatbot":
    st.title("💬 Chatbot")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    prompt = st.chat_input("Type a message...")
    if prompt and api_key:
        genai.configure(api_key=api_key)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        prompt_text = f"SYSTEM: Engage in a helpful conversation with the user, considering they have Global Developmental Delay.\nUSER: {prompt}"
        response = model.generate_content(prompt_text).text
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

# Practice Exam Generator Tab
elif tabs == "Practice Exam Generator":
    st.title("📝 Practice Exam Generator")
    difficulty = st.selectbox("Select Difficulty", ["Easy", "Medium", "Hard"])
    subject = st.text_input("Enter the subject", placeholder="e.g., Math, Science, History")
    topic = st.text_input("Enter a specific topic", placeholder="e.g., Algebra, Photosynthesis")
    
    if st.button("Generate Exam") and api_key and subject and topic:
        genai.configure(api_key=api_key)
        prompt_text = f"Generate {difficulty} level practice exam questions for {subject} on the topic of {topic}. Ensure the questions are suitable for students with Global Developmental Delay. Include 5 questions with 4 multiple-choice answers."
        response = model.generate_content(prompt_text).text
        st.write("### Practice Exam")
        st.write(response)

elif tabs == "Text Simplifier":
    st.title("📝 Text Simplifier")
    text_input = st.text_area("Enter the text you want to simplify", height=200)
    
    if st.button("Simplify Text") and api_key and text_input:
        try:
            genai.configure(api_key=api_key)
            prompt_text = f"Simplify the following text so that it is easy to understand for students with Global Developmental Delay and adapt based the inputted language:\n\n{text_input}"
            response = model.generate_content(prompt_text).text
            st.write("### Simplified Text")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred while simplifying the text: {e}")

