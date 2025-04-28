import streamlit as st
import google.generativeai as genai
import subprocess
import os
import tempfile
import textwrap
from typing import Optional, Tuple

# Constants
DEFAULT_CPP_FILE = "converted_program.cpp"
DEFAULT_EXE_FILE = "converted_program.exe"
MAX_CODE_LENGTH = 5000  # characters

def initialize_session_state():
    """Initialize all necessary session state variables."""
    if "translated_code" not in st.session_state:
        st.session_state.translated_code = ""
    if "compile_clicked" not in st.session_state:
        st.session_state.compile_clicked = False
    if "translation_success" not in st.session_state:
        st.session_state.translation_success = False
    if "compilation_output" not in st.session_state:
        st.session_state.compilation_output = ""

def configure_gemini(api_key: str) -> Optional[genai.GenerativeModel]:
    """Configure and return the Gemini model instance."""
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-flash-latest")
    except Exception as e:
        st.error(f"Failed to configure Gemini: {str(e)}")
        return None

def generate_translation_prompt(python_code: str) -> str:
    """Generate a detailed prompt for code translation."""
    return f"""
    Translate the following Python code to optimized, compilable C++ code:
    
    1. Preserve all functionality and logic
    2. Use modern C++ standards (C++17 or later)
    3. Handle memory management appropriately (RAII, smart pointers)
    4. Include necessary standard library headers
    5. Replace Python dynamic typing with appropriate C++ static types
    6. Convert Python standard library calls to equivalent C++ calls
    
    Python Code:
    ```python
    {python_code}
    ```
    
    Provide only the C++ code without additional explanations or markdown formatting.
    """

def translate_code(model: genai.GenerativeModel, python_code: str) -> Tuple[bool, str]:
    """Translate Python code to C++ using Gemini."""
    try:
        prompt = generate_translation_prompt(python_code)
        response = model.generate_content(prompt)
        
        if not response.text:
            return False, "Empty response from translation model"
            
        # Clean up the response
        clean_code = response.text.replace('```cpp', '').replace('```', '').strip()
        return True, clean_code
        
    except Exception as e:
        return False, f"Translation error: {str(e)}"

def compile_cpp_code(cpp_code: str, cpp_filename: str, exe_filename: str) -> Tuple[bool, str]:
    """Compile C++ code to executable and return status and output."""
    try:
        # Save to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            cpp_path = os.path.join(temp_dir, cpp_filename)
            exe_path = os.path.join(temp_dir, exe_filename)
            
            # Write C++ code to file
            with open(cpp_path, 'w') as f:
                f.write(cpp_code)
            
            # Compile with optimizations and warnings
            compile_cmd = [
                "g++", 
                "-std=c++17",
                "-Wall", 
                "-Wextra",
                "-O2",
                cpp_path,
                "-o", 
                exe_path
            ]
            
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                cwd=temp_dir
            )
            
            if result.returncode == 0:
                # Read the executable binary
                with open(exe_path, 'rb') as f:
                    exe_data = f.read()
                return True, exe_data
            else:
                return False, result.stderr
                
    except Exception as e:
        return False, f"Compilation error: {str(e)}"

def display_sidebar() -> str:
    """Render the sidebar and return the selected API key."""
    with st.sidebar:
        st.title("ConvertPy Settings")
        
        # API Key Input
        api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Enter your Google Gemini API key for code translation"
        )
        
        # About section
        st.markdown("---")
        st.markdown("### About ConvertPy")
        st.markdown("""
        ConvertPy leverages LLMs to translate Python code to C++ and compile it 
        to native executables, based on research by Al-Hussain & Alsubait (2025).
        """)
        
        return api_key

def display_home_page():
    """Render the home/landing page."""
    st.title("🐍 ConvertPy: Python to C++ Executable Translator")
    st.markdown("""
    ### Research-Based Code Translation
        
    Welcome to ConvertPy, an implementation of the LLM-powered Python-to-C++ translation 
    framework described in the paper:
    
    > **ConvertPy: Leveraging Large Language Models (LLMs) for Reliable and Scalable Code Translations**  
    > *Safa'a al-Hussain & Tahani Alsubait (2025)*
    
    ### Key Features:
    - **AI-Powered Translation**: Uses Google Gemini for accurate Python-to-C++ conversion
    - **Native Executables**: Compiles translated code to platform-specific binaries
    - **Research-Backed**: Implements the architecture from the ConvertPy paper
    
    ### How It Works:
    1. Enter your Python code in the translator tab
    2. The system will generate equivalent C++ code
    3. Compile the C++ code to a standalone executable
    4. Download and run the executable
    
    ### Supported Use Cases:
    - Converting Python prototypes to optimized C++ implementations
    - Creating distributable versions of Python tools
    - Educational comparisons between Python and C++ syntax
    - IoT and embedded systems development
    """)

def display_translator_page(api_key: str):
    """Render the code translation page."""
    st.title("Python to C++ Code Translator")
    
    # Code input section
    python_code = st.text_area(
        "Enter Python Code",
        height=300,
        placeholder="Enter your Python code here...",
        help="The Python code you want to translate to C++"
    )
    
    # Translation button
    if st.button("Translate to C++", disabled=not (api_key and python_code)):
        if len(python_code) > MAX_CODE_LENGTH:
            st.error(f"Code exceeds maximum length of {MAX_CODE_LENGTH} characters")
            return
            
        with st.spinner("Translating Python to C++..."):
            model = configure_gemini(api_key)
            if model:
                success, result = translate_code(model, python_code)
                
                if success:
                    st.session_state.translated_code = result
                    st.session_state.translation_success = True
                    st.session_state.compile_clicked = False
                    
                    st.success("Translation successful!")
                    st.subheader("Translated C++ Code")
                    st.code(result, language="cpp")
                else:
                    st.error(f"Translation failed: {result}")
    
    # Display translated code if available
    if st.session_state.translated_code:
        st.subheader("Translated C++ Code")
        st.code(st.session_state.translated_code, language="cpp")
        
        # Compilation section
        if st.button("Compile to Executable", disabled=not st.session_state.translation_success):
            st.session_state.compile_clicked = True
            
        if st.session_state.compile_clicked:
            with st.spinner("Compiling C++ code..."):
                success, result = compile_cpp_code(
                    st.session_state.translated_code,
                    DEFAULT_CPP_FILE,
                    DEFAULT_EXE_FILE
                )
                
                if success:
                    st.success("Compilation successful!")
                    st.download_button(
                        label="Download Executable",
                        data=result,
                        file_name=DEFAULT_EXE_FILE,
                        mime="application/octet-stream"
                    )
                else:
                    st.error("Compilation failed")
                    st.text_area("Compiler Output", result, height=200)

def main():
    """Main application flow."""
    # Initialize session state
    initialize_session_state()
    
    # Set page config
    st.set_page_config(
        page_title="ConvertPy - Python to C++ Translator",
        page_icon="🔄",
        layout="wide"
    )
    
    # Get API key from sidebar
    api_key = display_sidebar()
    
    # Page selection
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🔄 Translator"],
        label_visibility="collapsed"
    )
    
    # Display selected page
    if page == "🏠 Home":
        display_home_page()
    else:
        if not api_key:
            st.warning("Please enter your Google API key in the sidebar to use the translator")
        display_translator_page(api_key)

if __name__ == "__main__":
    main()
