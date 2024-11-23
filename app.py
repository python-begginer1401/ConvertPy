import streamlit as st
import google.generativeai as genai
from gtts import gTTS
from io import BytesIO
import textwrap
import subprocess  # For running system commands like g++ for C++ compilation
import os  # For handling file paths

# Initialize Google Gemini model
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Sidebar for API Key input and tab selection
with st.sidebar:
    st.sidebar.title("Navigation")
    tabs = st.sidebar.radio("Select an option", ["🏠 Home", "📝 Convert Python to Executable"])

    api_key = st.text_input("Google API Key", key="geminikey", type="password")

# Initialize session states for button clicks
if "translated_code" not in st.session_state:
    st.session_state["translated_code"] = ""
if "compile_clicked" not in st.session_state:
    st.session_state["compile_clicked"] = False

# Function to convert text to Markdown format
def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

# Function to compile C++ code to an executable
def compile_cpp_to_exe(cpp_code, file_name="program"):
    cpp_file_path = f"{file_name}.cpp"
    exe_file_path = f"{file_name}.exe"
    
    # Save the C++ code to a file
    with open(cpp_file_path, "w") as cpp_file:
        cpp_file.write(cpp_code)
    
    # Compile the C++ file using g++
    compile_command = ["g++", cpp_file_path, "-o", exe_file_path]
    try:
        result = subprocess.run(compile_command, capture_output=True, text=True)
        if result.returncode == 0:
            st.success(f"Compilation successful! You can download the executable below.")
            return exe_file_path
        else:
            st.error(f"Error during compilation:\n{result.stderr}")
            return None
    except subprocess.CalledProcessError as e:
        st.error(f"Error during compilation: {e}")
        return None
    finally:
        # Clean up the temporary C++ file
        if os.path.exists(cpp_file_path):
            os.remove(cpp_file_path)

# Main Page Tab
if tabs == "🏠 Home":
    st.title("🐍 ConvertPy")
    st.write("""
        Welcome to ConvertPy! 
        
        ConvertPy is an advanced AI-powered platform designed to effortlessly convert Python code into optimized C++ and compile it into executable files. 
        Select the tab from the sidebar to get started!
    """)

# Convert Python to Executable Tab
elif tabs == "📝 Convert Python to Executable":
    st.title("📝 Convert Python to Executable")
    text_input = st.text_area("Enter your Python code 🐍", height=200, placeholder="Write your Python code here...")

    # Translate Button
    if st.button("Translate Code to C++") and api_key and text_input:
        try:
            # Configure Google Gemini AI with API Key
            genai.configure(api_key=api_key)
            prompt_text = f"Translate the following Python code to equivalent C++ code ensure that the translated C++ code does not include Markdown formatting:\n\n{text_input}"
            response = model.generate_content(prompt_text).text

            # Remove Markdown formatting from the response
            clean_response = response.replace('```cpp', '').replace('```', '').strip()

            # Store translated code in session state
            st.session_state["translated_code"] = clean_response
            st.session_state["compile_clicked"] = False  # Reset compile state

            st.write("### Translated C++ Code")
            st.code(clean_response, language='cpp')

        except Exception as e:
            st.error(f"An error occurred during translation: {e}")

    # Display Compile Button if Translation Exists
    if st.session_state["translated_code"]:
        st.write("### Ready to Compile")
        if st.button("Compile C++ to Executable"):
            st.session_state["compile_clicked"] = True

    # Handle Compilation Process
    if st.session_state["compile_clicked"]:
        exe_file = compile_cpp_to_exe(st.session_state["translated_code"])
        if exe_file:
            with open(exe_file, "rb") as file:
                st.download_button(
                    label="Download Executable",
                    data=file,
                    file_name=exe_file,
                    mime="application/octet-stream"
                )
        else:
            st.error("Compilation failed. Check the C++ code or try again.")
