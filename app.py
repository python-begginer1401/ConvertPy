import streamlit as st
import google.generativeai as genai
import subprocess
import os
import platform
from io import BytesIO
import requests
import time

# --- Utility Functions ---
def ping_streamlit_app(app_url="https://convertpy.streamlit.app/", interval_seconds=300):
    """Pings a Streamlit app at a specified interval to prevent it from idling."""
    while True:
        try:
            requests.get(app_url)
            time.sleep(interval_seconds)
        except requests.exceptions.RequestException as e:
            print(f"Error pinging app: {e}")
            time.sleep(interval_seconds * 2) # Longer delay on error
        except KeyboardInterrupt:
            print("Pinging stopped by user.")
            break

def get_os_info():
    """Identifies the operating system and its executable extension."""
    system = platform.system().lower()
    if system == "darwin":
        return "macOS", ""
    elif system == "windows":
        return "Windows", ".exe"
    elif system == "linux":
        return "Linux", ""
    else:
        return system.capitalize(), ""

def save_code_to_file(code_string, file_path):
    """Safely saves a string of code to a file."""
    try:
        with open(file_path, "w") as file:
            file.write(code_string)
        return True
    except IOError as e:
        st.error(f"Error saving code to {file_path}: {e}")
        return False

def compile_cpp_code(cpp_code, file_name="program"):
    """Compiles C++ code for the detected operating system."""
    cpp_file_path = f"{file_name}.cpp"
    os_name, exe_ext = get_os_info()
    exe_name = f"{file_name}{exe_ext}"

    if not save_code_to_file(cpp_code, cpp_file_path):
        return None

    compile_command = ["g++", cpp_file_path, "-o", exe_name]
    if os_name != "Windows":
        compile_command.extend(["-std=c++11", "-pthread"])

    try:
        result = subprocess.run(compile_command, capture_output=True, text=True, check=True)
        st.success(f"Compilation successful for {os_name}!")
        return exe_name
    except subprocess.CalledProcessError as e:
        st.error(f"Compilation error:\n{e.stderr}")
        return None
    except FileNotFoundError:
        st.error(f"Compiler not found. Please ensure g++ is installed on your {os_name} system.")
        if os_name == "macOS":
            st.info("Install Xcode Command Line Tools: `xcode-select --install`")
        elif os_name == "Linux":
            st.info("Install build-essential: `sudo apt-get update && sudo apt-get install build-essential`")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during compilation: {e}")
        return None
    finally:
        if os.path.exists(cpp_file_path):
            os.remove(cpp_file_path)

def translate_python_to_cpp(python_code, api_key, model_name):
    """Translates Python code to C++ using the specified AI model."""
    if not api_key:
        st.error(f"Please enter your API Key for {model_name} in the sidebar.")
        return None

    try:
        if model_name.startswith("gemini"):
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)

            prompt = f"""Convert the following Python code to standard, portable C++11 code:
            ```python
            {python_code}
            ```

            Requirements:
            1. Adhere strictly to the C++11 standard for maximum portability.
            2. Ensure compatibility across Windows, Linux, and macOS.
            3. Include all necessary standard library headers.
            4. Provide only the raw C++ code, without any additional explanations or comments.
            5. Maintain proper indentation and code formatting.
            6. Implement input/output handling using standard C++ streams where necessary.
            7. Employ modern C++ practices for clarity and efficiency.
            8. Avoid any platform-specific libraries or APIs.
            """

            response = model.generate_content(prompt)

            if response and hasattr(response, 'text'):
                cpp_code = response.text.strip()
            elif response.candidates and response.candidates[0].content.parts:
                cpp_code = response.candidates[0].content.parts[0].text.strip()
            else:
                st.error(f"Failed to retrieve a valid C++ code response from the {model_name} model.")
                return None

            # Clean up potential markdown formatting
            if cpp_code.startswith("```cpp"):
                cpp_code = cpp_code[6:]
            if cpp_code.endswith("```"):
                cpp_code = cpp_code[:-3]
            return cpp_code.strip()
        elif model_name.startswith("huggingface"):
            st.error("Hugging Face integration is not yet implemented in this version.")
            st.info("Please check for future updates.")
            return None
        else:
            st.error(f"Unsupported AI model: {model_name}")
            return None

    except Exception as e:
        st.error(f"Code translation failed with {model_name}: {e}")
        st.error("Please verify your API key and model name.")
        return None

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="ConvertPy", page_icon="🐍")

    # Initialize session states
    if "translated_code" not in st.session_state:
        st.session_state["translated_code"] = ""
    if "compile_clicked" not in st.session_state:
        st.session_state["compile_clicked"] = False

    # Sidebar for navigation and API key/model selection
    with st.sidebar:
        st.sidebar.title("⚙️ Configuration")
        tabs = st.sidebar.radio("Navigate", ["🏠 Home", "📝 Convert & Compile"])
        ai_model_option = st.selectbox("Choose AI Model", ["gemini-1.5-pro-latest", "gemini-pro", "huggingface (Not Implemented Yet)"])

        api_key = None
        if ai_model_option.startswith("gemini"):
            api_key = st.text_input(f"🔑 Google API Key for {ai_model_option}", key="geminikey", type="password", help="Enter your Google Gemini API key")
        elif ai_model_option.startswith("huggingface"):
            api_key = st.text_input(f"🔑 Hugging Face API Token", key="huggingface_token", type="password", help="Enter your Hugging Face API token (if required)")

    # Main content area
    if tabs == "🏠 Home":
        st.title("🐍 ConvertPy: Python to Executable")
        os_name, _ = get_os_info()
        st.markdown(
            f"""
            Welcome to **ConvertPy**, a tool designed to translate your Python code into portable C++ and compile it into an executable for your system.

            **Detected Operating System:** `{os_name}`

            **Key Features:**
            - Translate Python code to standard C++11 using your choice of AI model (currently supports Google Gemini).
            - Compile the generated C++ code directly within the application.
            - Download the compiled executable for your operating system.
            - Supports Windows, Linux, and macOS.

            **Prerequisites:**
            - An API Key for your chosen AI model (enter in the sidebar).
            - The `g++` compiler must be installed on your system.

            **Get Started:**
            1. Navigate to the "Convert & Compile" tab in the sidebar.
            2. Choose your preferred AI model from the dropdown.
            3. Enter the corresponding API key in the sidebar.
            4. Enter your Python code in the text area.
            5. Click "Translate Code to C++".
            6. Review the translated C++ code.
            7. Click "Compile for My System" to build the executable.
            8. Download the generated executable.

            **Note:** The accuracy and functionality of the translated code depend on the complexity and structure of the input Python code and the capabilities of the AI model.
            """
        )
        st.info("For optimal results, ensure your Python code is well-structured and adheres to standard practices.")

    elif tabs == "📝 Convert & Compile":
        st.title("📝 Python to Executable Conversion")
        os_name, exe_ext = get_os_info()
        st.info(f"Detected Operating System: **{os_name}**")

        python_code = st.text_area("Enter your Python code here:", height=250, placeholder="```python\nprint('Hello, World!')\n```")

        if st.button("✨ Translate Code to C++"):
            if python_code:
                with st.spinner(f"Translating with {ai_model_option}..."):
                    cpp_code = translate_python_to_cpp(python_code, st.session_state.get("geminikey") if ai_model_option.startswith("gemini") else st.session_state.get("huggingface_token"), ai_model_option)
                    if cpp_code:
                        st.session_state["translated_code"] = cpp_code
                        st.session_state["compile_clicked"] = False
                        st.subheader("Generated C++ Code:")
                        st.code(st.session_state["translated_code"], language='cpp')
            else:
                st.warning("Please enter Python code to translate.")

        if st.session_state.get("translated_code"):
            st.subheader("Compilation")
            if st.button("🔨 Compile for My System"):
                st.session_state["compile_clicked"] = True

        if st.session_state.get("compile_clicked"):
            with st.spinner("Compiling C++ code..."):
                executable_file = compile_cpp_code(st.session_state["translated_code"])
                if executable_file:
                    try:
                        with open(executable_file, "rb") as f:
                            executable_bytes = f.read()
                        st.download_button(
                            label=f"💾 Download Executable for {os_name}",
                            data=executable_bytes,
                            file_name=f"program{exe_ext}",
                            mime="application/octet-stream",
                            key="download_button"
                        )
                    except Exception as e:
                        st.error(f"Error preparing the executable for download: {e}")
                    finally:
                        if os.path.exists(executable_file):
                            os.remove(executable_file)

if __name__ == "__main__":
    # Start the pinging process in a separate thread
    import threading
    ping_thread = threading.Thread(target=ping_streamlit_app, daemon=True)
    ping_thread.start()

    main()
