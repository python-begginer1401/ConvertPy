

import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from gtts import gTTS
from io import BytesIO
import textwrap


# Initialize Google Gemini model
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Sidebar for API Key input and tab selection
with st.sidebar:
    tabs = st.sidebar.radio("Services/Programs", ["🏠 Home", "💬 Chatbot Specialist", "📝 File Q&A", "🎧 Audio Explanation Generator", "📚 Practice Exam Generator", "📝 Text Simplifier" ])


    api_key = st.text_input("Google API Key", key="geminikey", type="password")

def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)


# Main Page Tab
if tabs == "🏠 Home":
    st.title("🌬️ Ease Platform for Speacial Students ")
    st.write("""
        Welcome to the AI tools Platform! 
        
        - 📝 **File Q&A**: Upload an article and get answers to your questions in a simplified manner.
        - 💬 **Chatbot Specialist**: Interact with a chatbot with any help you might need or if you just want to chat.
        - 📚 **Practice Exam Generator**: Generate practice exams based on difficulty, subject, and topic to aid learning.
        - 🎧**Audio Explanation Generator**: Generate audio explanations with text-to-speech functionality for better understanding.
        - 📝**Text Simplifier**: Simplify text to make it more accessible and easier to understand.

        Select a tab from the sidebar to get started!
    """)

# Audio Explanation Generator Tab
elif tabs == "🎧 Audio Explanation Generator":
    
    st.title("🎧 Audio Explanation Generator")

    lesson_subject = st.text_input("Enter the lesson subject", placeholder="e.g., Math, Science, History")
    lesson_topic = st.text_input("Enter the lesson topic", placeholder="e.g., Algebra, Photosynthesis, World War II")

    if st.button("Generate Lesson"):
        try:
            prompt = f"Write a simple lesson about {lesson_topic} in the subject of {lesson_subject} and ensure that it is written in a simple mannar targeted towards students with global developmental delay (gdd)"
            response = model.generate_content(prompt)
            lesson_text = response.text

            if lesson_text:
                st.write("### Lesson")
                st.write(to_markdown(lesson_text))

                tts = gTTS(text=lesson_text, lang='en')
                audio_path = "lesson_audio.mp3"
                tts.save(audio_path)

                # Display the audio to the user
                audio_file = open(audio_path, "rb").read()
                st.audio(audio_file, format="audio/mp3")
                st.download_button("Download Lesson Audio", data=audio_file, file_name="lesson_audio.mp3", mime="audio/mp3")
            else:
                st.warning("No output generated. Please try again with a different lesson subject or topic.")
        except Exception as e:
            st.error(f"An error occurred while generating the lesson: {e}")
# File Q&A Tab
elif tabs == "📝 File Q&A":
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
elif tabs == "💬 Chatbot Specialist":
    st.title("💬 Chatbot Specialist")
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
elif tabs == "📚 Practice Exam Generator":
    st.title("📚 Practice Exam Generator")
    difficulty = st.selectbox("Select Difficulty", ["Easy", "Medium", "Hard"])
    subject = st.text_input("Enter the subject", placeholder="e.g., Math, Science, History")
    topic = st.text_input("Enter a specific topic", placeholder="e.g., Algebra, Photosynthesis")
    
    if st.button("Generate Exam") and api_key and subject and topic:
        genai.configure(api_key=api_key)
        prompt_text = f"Generate {difficulty} level practice exam questions for {subject} on the topic of {topic}. Ensure the questions are suitable for students with Global Developmental Delay. Ensure that you generate multiple question types also use very simple vocabulary since the student using this has GDD global developmental delay also add the answers at the end and ensure that the questions are based on the difficultiy entered "
        response = model.generate_content(prompt_text).text
        st.write("### Practice Exam")
        st.write(response)

elif tabs == "📝 Text Simplifier":
    st.title("📝 Text Simplifier")
    text_input = st.text_area("Enter the text you want to simplify", height=200)
    
    if st.button("Simplify Text") and api_key and text_input:
        try:
            genai.configure(api_key=api_key)
            prompt_text = f"Simplify the following text so that it is easy to understand for students with Global Developmental Delay and respond with the inputted language:\n\n{text_input}"
            response = model.generate_content(prompt_text).text
            st.write("### Simplified Text")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred while simplifying the text: {e}")
