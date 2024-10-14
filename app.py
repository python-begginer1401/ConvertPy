import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from gtts import gTTS
from io import BytesIO
import textwrap
import subprocess  # To run system commands like g++ for C++ compilation
import os  # To handle file paths

# Initialize Google Gemini model
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Sidebar for API Key input and tab selection
with st.sidebar:
    st.sidebar.title("Navigation")
    tabs = st.sidebar.radio("Select an option", ["🏠 Home", "📝 Convert python to executable"])

    api_key = st.text_input("Google API Key", key="geminikey", type="password")

def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

# Function to compile C++ code to an executable
def compile_cpp_to_exe(cpp_code, file_name="program"):
    # Save the C++ code to a file
    cpp_file_path = f"{file_name}.cpp"
    with open(cpp_file_path, "w") as cpp_file:
        cpp_file.write(cpp_code)

    # Compile the C++ file using g++
    exe_file_path = f"{file_name}.exe"
    compile_command = ["g++", cpp_file_path, "-o", exe_file_path]
    try:
        subprocess.run(compile_command, check=True)
        st.success(f"Successfully compiled! You can download the executable below.")
        return exe_file_path
    except subprocess.CalledProcessError as e:
        st.error(f"Error compiling C++ code: {e}")
        return None

# Main Page Tab
if tabs == "🏠 Home":
    st.title("🐍 ConvertPy")
    st.write("""
        Welcome to ConvertPy! 
        
        ConvertPy is an advanced AI-powered platform designed to effortlessly convert Python code into optimized C++ and compile it into executable files. It streamlines the process of cross-platform compatibility by offering seamless translation, ensuring the integrity of the original code, and allowing developers to focus on performance. Whether you're a programmer looking for efficiency or a developer seeking better control over executable files, ConvertPy offers a fast, reliable solution with cutting-edge AI translation technology.

        Select the tab from the sidebar to get started!
    """)

elif tabs == "📝 Convert python to executable":
    st.title("📝 Convert python to executable")
    text_input = st.text_area("Enter your Python code 🐍 ", height=200, placeholder="Write your Python code here...")

    if st.button("Translate code to C++") and api_key and text_input:
        try:
            # Configure Google Gemini AI with API Key
            genai.configure(api_key=api_key)
            prompt_text = f"You are a code translator. Translate the following Python code to C++:\n\n{text_input}"
            response = model.generate_content(prompt_text).text
            st.write("### Translated C++ Code")
            st.code(response, language='cpp')

            # Button to compile C++ code to executable
            if st.button("Convert C++ to Executable"):
                exe_file = compile_cpp_to_exe(response)  # Pass the translated C++ code
                if exe_file:
                    with open(exe_file, "rb") as file:
                        btn = st.download_button(
                            label="Download Executable",
                            data=file,
                            file_name=f"{exe_file}",
                            mime="application/octet-stream"
                        )

        except Exception as e:
            st.error(f"An error occurred while translating the code: {e}")
