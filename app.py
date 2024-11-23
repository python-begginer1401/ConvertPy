import streamlit as st
from io import BytesIO
import textwrap
import subprocess
import os

try:
    import google.generativeai as genai
except ImportError:
    st.error("Please install `google-generativeai` library using `pip install google-generativeai`.")

# Sidebar for API Key input and navigation
with st.sidebar:
    st.title("Navigation")
    tabs = st.radio("Select an option", ["🏠 Home", "📝 Convert Python to Executable"])

    api_key = st.text_input("Google API Key", type="password")

# Initialize session states for handling actions
if "translated_code" not in st.session_state:
    st.session_state["translated_code"] = ""
if "compile_clicked" not in st.session_state:
    st.session_state["compile_clicked"] = False

# Helper function to compile C++ code into a 64-bit executable
def compile_cpp_to_exe(cpp_code, file_name="program"):
    cpp_file_path = os.path.abspath(f"{file_name}.cpp")
    exe_file_path = os.path.abspath(f"{file_name}.exe")

    # Save the C++ code to a file
    with open(cpp_file_path, "w") as cpp_file:
        cpp_file.write(cpp_code)

    # Compile using g++ with 64-bit architecture
    compile_command = ["g++", "-m64", cpp_file_path, "-o", exe_file_path]
    try:
        result = subprocess.run(compile_command, capture_output=True, text=True)
        if result.returncode == 0:
            st.success(f"Compilation successful! Download your executable below.")
            return exe_file_path
        else:
            st.error(f"Compilation failed:\n{result.stderr}")
            return None
    except Exception as e:
        st.error(f"Error during compilation: {e}")
        return None
    finally:
        # Clean up the temporary C++ file
        if os.path.exists(cpp_file_path):
            os.remove(cpp_file_path)

# Main Home Tab
if tabs == "🏠 Home":
    st.title("🐍 ConvertPy")
    st.write("""
        Welcome to ConvertPy! 
        
        Convert Python code into C++ and compile it to an executable. Get started by navigating to the relevant tab.
    """)

# Convert Python to Executable Tab
elif tabs == "📝 Convert Python to Executable":
    st.title("📝 Convert Python to Executable")
    python_code = st.text_area("Enter your Python code", height=200, placeholder="Write your Python code here...")

    if st.button("Translate Code to C++") and api_key:
        try:
            # Configure Google Gemini AI
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash-latest")

            # Request translation
            prompt = f"Translate the following Python code to C++:\n\n{python_code}"
            response = model.generate_content(prompt_text=prompt)

            # Process and display the response
            cplusplus_code = response.text.replace("```cpp", "").replace("```", "").strip()
            st.session_state["translated_code"] = cplusplus_code

            st.write("### Translated C++ Code")
            st.code(cplusplus_code, language="cpp")
            st.session_state["compile_clicked"] = False  # Reset compile state

        except Exception as e:
            st.error(f"Error during translation: {e}")

    if st.session_state["translated_code"]:
        if st.button("Compile C++ to Executable"):
            st.session_state["compile_clicked"] = True

    if st.session_state["compile_clicked"]:
        executable_file = compile_cpp_to_exe(st.session_state["translated_code"])
        if executable_file:
            with open(executable_file, "rb") as file:
                st.download_button(
                    label="Download Executable",
                    data=file,
                    file_name=os.path.basename(executable_file),
                    mime="application/octet-stream"
                )
        else:
            st.error("Compilation failed. Check the C++ code or retry.")
