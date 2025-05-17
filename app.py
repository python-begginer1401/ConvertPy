import streamlit as st
from streamlit_option_menu import option_menu
import requests
import threading
import time
import google.generativeai as genai
import subprocess
import os
import platform
from typing import Optional

# Set page configuration
st.set_page_config(
    page_title="ConvertPy",
    page_icon="🐍",
    layout="wide"
)

# URL for self-pinging
APP_URL = "https://convertpy2025.streamlit.app/"

# Self-pinging function
def keep_awake(url):
    while True:
        try:
            requests.get(url)
        except Exception:
            pass
        time.sleep(600)

# Start keep-alive thread
threading.Thread(target=keep_awake, args=(APP_URL,), daemon=True).start()

# Compile C++ code with enhanced cross-platform support
def compile_cpp_to_exe(cpp_code: str, file_name: str = "program") -> Optional[str]:
    # Determine OS-specific settings
    system = platform.system().lower()
    
    # Set up OS-specific configurations
    configs = {
        "windows": {
            "compiler": "g++",
            "ext": ".exe",
            "extra_flags": [],
            "error_msg": "Make sure MinGW or Visual Studio Build Tools is installed"
        },
        "darwin": {  # macOS
            "compiler": "clang++",
            "ext": "",
            "extra_flags": ["-std=c++17"],
            "error_msg": "Make sure Xcode Command Line Tools are installed (run: xcode-select --install)"
        },
        "linux": {
            "compiler": "g++",
            "ext": "",
            "extra_flags": ["-std=c++17"],
            "error_msg": "Make sure build-essential package is installed (run: sudo apt-get install build-essential)"
        }
    }

    # Get OS-specific configuration
    config = configs.get(system, configs["linux"])  # Default to Linux config if OS not recognized
    
    cpp_file_path = f"{file_name}.cpp"
    exe_file_path = file_name + config["ext"]

    # Prepare compilation command with OS-specific flags
    compile_command = [config["compiler"], cpp_file_path, "-o", exe_file_path] + config["extra_flags"]

    try:
        # Write the C++ code to a file
        with open(cpp_file_path, "w", encoding="utf-8") as cpp_file:
            cpp_file.write(cpp_code)

        # Try to compile
        result = subprocess.run(compile_command, capture_output=True, text=True)
        
        if result.returncode == 0:
            st.success("✅ Compilation successful! Download your executable below.")
            return exe_file_path
        else:
            error_msg = f"❌ Compilation failed:\n{result.stderr}\n\n{config['error_msg']}"
            st.error(error_msg)
            return None

    except FileNotFoundError:
        st.error(f"❌ Compiler not found. {config['error_msg']}")
        return None
    except Exception as e:
        st.error(f"❌ Error during compilation: {str(e)}")
        return None
    finally:
        # Clean up the source file
        if os.path.exists(cpp_file_path):
            try:
                os.remove(cpp_file_path)
            except:
                pass

# Track recent activity
def add_recent_activity(activity: str):
    if "recent_activities" not in st.session_state:
        st.session_state["recent_activities"] = []
    st.session_state["recent_activities"].append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {activity}")
    st.session_state["recent_activities"] = st.session_state["recent_activities"][-5:]

# Home page
def home_page():
    st.markdown("# 🐍 Welcome to ConvertPy!")
    with st.container():
        st.markdown(
            """
            ### About ConvertPy
            A powerful Python to C++ code converter with cross-platform support.
            
            #### Features:
            - 🔄 Convert Python code to optimized C++
            - 🖥️ Cross-platform compilation support
            - 🤖 Multiple AI models for translation
            - 📝 Detailed documentation and examples
            
            #### Latest Update
            Last updated: 2025-05-16 23:57:23 UTC
            
            #### Getting Started
            1. Navigate to the "Convert" tab
            2. Choose your preferred AI model
            3. Enter your code
            4. Get your executable!
            """
        )

# Convert page
def convert_page():
    st.markdown("# 📝 Convert Python to Executable")

    # Step 1: Model and API Key
    with st.container():
        with st.expander("### Step 1: Choose Model", expanded=True):
            model_selection = st.selectbox(
                "Select a model",
                ["Gemini", "Mistral-small", "DeepSeek", "Llama"],
                key="model_choice"
            )
            api_key = st.text_input(f"Enter API Key for {model_selection}", type="password")

    # Step 2: Input Python Code
    text_input = None
    with st.container():
        with st.expander("### Step 2: Provide Python Code", expanded=False):
            input_method = st.radio(
                "Choose Input Method",
                ["Type Code", "Upload File"],
                key="input_method"
            )
            if input_method == "Type Code":
                text_input = st.text_area(
                    "Enter your Python code 🐍",
                    height=200,
                    placeholder="Write your Python code here..."
                )
            elif input_method == "Upload File":
                uploaded_file = st.file_uploader("Upload your Python file", type=["py"])
                if uploaded_file:
                    text_input = uploaded_file.read().decode("utf-8")

    # Step 3: Translate and Compile
    if text_input:
        with st.container():
            with st.expander("### Step 3: Translate and Compile", expanded=False):
                if st.button("Translate to C++"):
                    try:
                        prompt = (
                            "You are a code translation expert. "
                            "Translate the following Python code to a complete, compilable C++ program. "
                            "Include necessary headers, a main() function, and provide only the raw C++ code without any explanations or markdown fences.\n\n" + text_input
                        )
                        
                        translated_code = None
                        
                        if model_selection == "Gemini":
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel("gemini-1.5-flash-latest")
                            response = model.generate_content(prompt)
                            translated_code = response.text

                        elif model_selection == "Mistral-small":
                            headers = {
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json"
                            }
                            payload = {
                                "model": "mistral-small",
                                "messages": [
                                    {"role": "system", "content": prompt},
                                    {"role": "user", "content": ""}
                                ],
                                "temperature": 0,
                                "max_tokens": 2000
                            }
                            response = requests.post(
                                "https://api.mistral.ai/v1/chat/completions",
                                headers=headers,
                                json=payload,
                                timeout=30
                            )
                            response_data = response.json()
                            if 'choices' in response_data and len(response_data['choices']) > 0:
                                translated_code = response_data['choices'][0]['message']['content']

                        elif model_selection in ["DeepSeek", "Llama"]:
                            headers = {"Authorization": f"Bearer {api_key}"}
                            model_id = (
                                "deepseek-ai/deepseek-llm"
                                if model_selection == "DeepSeek"
                                else "meta-llama/llama-3.3-8b-instruct:free"
                            )
                            payload = {
                                "model": model_id,
                                "messages": [
                                    {"role": "system", "content": prompt},
                                    {"role": "user", "content": ""}
                                ]
                            }
                            response = requests.post(
                                "https://openrouter.ai/api/v1/chat/completions",
                                json=payload,
                                headers=headers,
                                timeout=30
                            )
                            response_data = response.json()
                            if 'choices' in response_data and len(response_data['choices']) > 0:
                                translated_code = response_data['choices'][0]['message']['content']

                        if translated_code:
                            translated_code = translated_code.replace("```cpp", "").replace("```", "").strip()
                            st.session_state["translated_code"] = translated_code
                            st.markdown("### Translated C++ Code")
                            st.code(translated_code, language="cpp")
                            add_recent_activity(f"Translated Python code to C++ using {model_selection}.")
                        else:
                            st.error("Translation failed: Empty response from API")

                    except Exception as e:
                        st.error(f"Translation failed: {str(e)}")

    # Step 4: Compile the C++ Code
    if st.session_state.get("translated_code"):
        with st.container():
            with st.expander("### Step 4: Compile and Download Executable", expanded=False):
                if st.button("Compile to Executable"):
                    exe_file = compile_cpp_to_exe(st.session_state["translated_code"])
                    if exe_file:
                        with open(exe_file, "rb") as file:
                            st.download_button(
                                label="Download Executable",
                                data=file,
                                file_name=os.path.basename(exe_file),
                                mime="application/octet-stream"
                            )
                        add_recent_activity("Compiled a C++ executable")

# Enhanced Help page
def help_page():
    st.markdown("# ❓ Help & Documentation")
    tabs = st.tabs(["Getting Started", "FAQ", "Troubleshooting", "API Keys", "Examples", "System Requirements"])

    with tabs[0]:
        st.markdown("""
        ### Getting Started
        1. **Select a Model and Enter API Key**
           - Choose from Gemini, Mistral-small, DeepSeek, or Llama
           - Enter your API key for the selected model
           - API keys are never stored and are used only for translation
        
        2. **Input Your Python Code**
           - Either type your code directly or upload a .py file
           - Code should be complete and runnable
           - Avoid using external dependencies when possible
        
        3. **Translate to C++**
           - Click "Translate to C++" to convert your code
           - Review the generated C++ code
           - The translation maintains Python functionality while optimizing for C++
        
        4. **Compile and Download**
           - Click "Compile to Executable" to create your program
           - Download the executable file
           - The program will be compiled for your current operating system
        """)

    with tabs[1]:
        st.markdown("""
        ### FAQ
        #### Models
        - **Gemini**: Best overall performance, especially for complex code
        - **Mistral-small**: Good balance of speed and accuracy
        - **DeepSeek/Llama**: Alternative options via OpenRouter
        
        #### Code Limitations
        - Maximum code size: ~2000 tokens
        - Supported Python features:
          - Basic control structures (if, while, for)
          - Functions and classes
          - Standard library modules
          - Basic file operations
        
        #### Operating System Support
        - Windows: Requires MinGW or Visual Studio Build Tools
        - macOS: Requires Xcode Command Line Tools
        - Linux: Requires GCC/G++
        
        #### Security
        - API keys are never stored
        - Code is processed locally for compilation
        - No data is retained between sessions
        """)

    with tabs[2]:
        st.markdown("""
        ### Troubleshooting
        #### Translation Errors
        - **Invalid API Key**: Verify your API key is correct and active
        - **Code Too Long**: Break down into smaller functions
        - **Syntax Errors**: Ensure Python code is valid before translation
        - **Timeout Issues**: Try reducing code complexity
        
        #### Compilation Errors
        - **Missing Compiler**:
          - Windows: Install MinGW or Visual Studio Build Tools
          - macOS: Run `xcode-select --install`
          - Linux: Install build-essential package
        
        - **Common Error Messages**:
          - "'g++' not found": Compiler not installed
          - "undefined reference": Missing function implementation
          - "permission denied": Check file system permissions
        
        #### Runtime Issues
        - **File Paths**: Use forward slashes (/) for paths
        - **Memory Management**: Check for memory leaks in C++ code
        - **Input/Output**: Ensure proper stream handling
        """)

    with tabs[3]:
        st.markdown("""
        ### API Keys
        #### How to Get API Keys
        1. **Gemini**
           - Visit [Google AI Studio](https://makersuite.google.com/)
           - Create an account and generate API key
           - Free tier available with limitations
        
        2. **Mistral**
           - Go to [Mistral AI](https://mistral.ai/)
           - Sign up for an account
           - Generate API key from dashboard
        
        3. **DeepSeek/Llama (via OpenRouter)**
           - Visit [OpenRouter.ai](https://openrouter.ai/)
           - Create account and verify email
           - Generate API key from settings
        
        #### API Key Security
        - Never share your API keys
        - Rotate keys periodically
        - Monitor usage for unauthorized access
        """)

    with tabs[4]:
        st.markdown("""
        ### Examples
        #### Basic Example
        ```python
        # Python input
        def factorial(n):
            if n <= 1:
                return 1
            return n * factorial(n-1)
        
        print(factorial(5))
        ```
        
        #### File Handling Example
        ```python
        # Python input
        with open('input.txt', 'r') as f:
            data = f.readlines()
            
        for line in data:
            print(line.strip())
        ```
        
        #### Class Example
        ```python
        # Python input
        class Rectangle:
            def __init__(self, width, height):
                self.width = width
                self.height = height
                
            def area(self):
                return self.width * self.height
        ```
        """)

    with tabs[5]:
        st.markdown("""
        ### System Requirements
        #### Windows
        - Windows 10/11
        - MinGW-w64 or Visual Studio Build Tools
        - Python 3.8 or higher
        - 4GB RAM minimum
        
        #### macOS
        - macOS 10.15 or higher
        - Xcode Command Line Tools
        - Python 3.8 or higher
        - 4GB RAM minimum
        
        #### Linux
        - Ubuntu 20.04 or equivalent
        - GCC/G++ 9.0 or higher
        - Python 3.8 or higher
        - 4GB RAM minimum
        
        #### Network
        - Stable internet connection
        - Access to API endpoints
        """)

# Recent Activity page
def recent_activity_page():
    st.markdown("# 🕒 Recent Activity")
    if "recent_activities" in st.session_state and st.session_state["recent_activities"]:
        for activity in st.session_state["recent_activities"]:
            st.markdown(f"- {activity}")
    else:
        st.markdown("No recent activity yet.")

# Main app
def main():
    with st.sidebar:
        st.markdown("## 🐍 **ConvertPy**")
        st.markdown("### Python to C++ Converter")
        st.markdown("---")
        
        choice = option_menu(
            menu_title="Navigation",
            options=["Home", "Convert", "Help", "Recent Activity"],
            icons=["house", "code-slash", "question-circle", "clock"],
            menu_icon="menu-button",
            default_index=0,
        )
        
        st.markdown("---")
        st.markdown("_Created by python-begginer1401 - 2025_", help="Contact: @SWE_ME on Telegram")

    if choice == "Home":
        home_page()
    elif choice == "Convert":
        convert_page()
    elif choice == "Help":
        help_page()
    elif choice == "Recent Activity":
        recent_activity_page()

if __name__ == "__main__":
    main()
