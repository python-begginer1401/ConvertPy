import streamlit as st
from streamlit_option_menu import option_menu
import requests  # For HTTP requests (self-pinging)
import threading  # For creating a background thread
import time
import google.generativeai as genai
import subprocess
import os
import platform
from typing import Optional

# Set page title and favicon
st.set_page_config(
    page_title="ConvertPy",  # Title of the browser tab
    page_icon="🐍",          # Emoji or path to an icon
    layout="wide"            # Optional: Makes the layout wider
)

# URL of the deployed app (self-pinging)
APP_URL = "https://convertpy2025.streamlit.app/"

# Self-pinging function to keep the app awake
def keep_awake(url):
    """
    Periodically pings the app's own URL to prevent it from sleeping.
    :param url: The URL of the deployed app.
    """
    while True:
        try:
            print(f"Pinging {url} to keep the app awake...")
            requests.get(url)
        except Exception as e:
            print(f"Failed to ping {url}: {e}")
        time.sleep(600)  # Ping every 10 minutes

# Start self-pinging in a background thread
threading.Thread(target=keep_awake, args=(APP_URL,), daemon=True).start()

# Compile C++ code into a portable executable
def compile_cpp_to_exe(cpp_code: str, file_name: str = "program", target_os: Optional[str] = None) -> Optional[str]:
    cpp_file_path = f"{file_name}.cpp"
    exe_file_path = file_name + (".exe" if platform.system() == "Windows" else "")

    compile_command = ["g++", cpp_file_path, "-o", exe_file_path]
    if target_os:
        if target_os.lower() == "windows":
            compile_command += ["-static", "-static-libgcc", "-static-libstdc++"]
        elif target_os.lower() == "linux":
            compile_command += ["-fPIC"]
        elif target_os.lower() == "macos":
            compile_command += ["-target", "x86_64-apple-darwin", "-stdlib=libc++"]

    try:
        with open(cpp_file_path, "w") as cpp_file:
            cpp_file.write(cpp_code)

        result = subprocess.run(compile_command, capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Compilation successful! Download your executable below.")
            return exe_file_path
        else:
            st.error(f"Compilation failed:\n{result.stderr}")
            return None
    except Exception as e:
        st.error(f"Error during compilation: {e}")
        return None
    finally:
        if os.path.exists(cpp_file_path):
            os.remove(cpp_file_path)

# Function to track recent activity
def add_recent_activity(activity: str):
    if "recent_activities" not in st.session_state:
        st.session_state["recent_activities"] = []
    st.session_state["recent_activities"].append(activity)
    # Keep only the last 5 activities
    st.session_state["recent_activities"] = st.session_state["recent_activities"][-5:]

# Display the Home page
def home_page():
    st.markdown("# 🐍 Welcome to ConvertPy!")
    with st.container():
        st.markdown(
            """
            ### About ConvertPy
            ConvertPy is a tool to translate Python code to C++ executables.
            - **Use the "Convert" tab to get started.**
            - **Supports generating executables for Windows, Linux, and macOS.**
            """
        )

# Display the Convert page with steps in collapsible cards
def convert_page():
    st.markdown("# 📝 Convert Python to Executable")

    # Step 1: Model and API Key
    with st.container():
        with st.expander("### Step 1: Choose Model", expanded=True):
            model_selection = st.selectbox(
                "Select a model",
                ["Gemini", "Hugging Face", "Claude", "Llama"],
                key="model_choice"
            )
            api_key = None
            if model_selection:
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
    translated_code = None
    if text_input:
        with st.container():
            with st.expander("### Step 3: Translate and Compile", expanded=False):
                if st.button("Translate to C++"):
                    try:
                        if model_selection == "Gemini":
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel("gemini-1.5-flash-latest")
                            prompt = f"Translate the following Python code to equivalent C++ code:\n\n{text_input}"
                            response = model.generate_content(prompt)
                            translated_code = response.text.replace("```cpp", "").replace("```", "").strip()

                        elif model_selection == "Hugging Face":
                            headers = {"Authorization": f"Bearer {api_key}"}
                            payload = {"inputs": f"Translate this Python code to C++:\n\n{text_input}"}
                            response = requests.post(
                                "https://api-inference.huggingface.co/models/your-model-name",
                                headers=headers,
                                json=payload
                            )
                            if response.status_code == 200:
                                translated_code = response.json().get("generated_text", "").strip()
                            else:
                                st.error(f"Translation failed: {response.json().get('error')}")

                        elif model_selection == "Claude":
                            st.error("Claude support is not yet implemented.")
                            return

                        elif model_selection == "Llama":
                            url = "https://api.llama.ai/v1/translate"
                            headers = {"Authorization": f"Bearer {api_key}"}
                            payload = {"text": text_input, "target_language": "C++"}
                            response = requests.post(url, headers=headers, json=payload)
                            if response.status_code == 200:
                                translated_code = response.json().get("translated_text", "").strip()
                            else:
                                st.error(f"Translation failed: {response.json().get('error')}")

                        st.session_state["translated_code"] = translated_code
                        st.markdown("### Translated C++ Code")
                        st.code(translated_code, language="cpp")
                        add_recent_activity(f"Translated Python code to C++ using {model_selection}.")
                    except Exception as e:
                        st.error(f"Translation failed: {e}")

    # Step 4: Compile the C++ Code
    if st.session_state.get("translated_code"):
        with st.container():
            with st.expander("### Step 4: Compile and Download Executable", expanded=False):
                target_os = st.selectbox("Compile for which OS?", ["Current OS", "Windows", "Linux", "macOS"])
                if st.button("Compile to Executable"):
                    exe_file = compile_cpp_to_exe(
                        st.session_state["translated_code"],
                        target_os=(None if target_os == "Current OS" else target_os.lower())
                    )
                    if exe_file:
                        with open(exe_file, "rb") as file:
                            st.download_button(
                                label="Download Executable",
                                data=file,
                                file_name=os.path.basename(exe_file),
                                mime="application/octet-stream"
                            )
                        add_recent_activity(f"Compiled a C++ executable for {target_os}.")

# Display the Help page
def help_page():
    st.markdown("# ❓ Help")
    st.markdown("""
    - **How to Use:** Use the "Convert" tab to translate and compile Python code.
    - **Supported OS:** Windows, Linux, macOS.
    - **Troubleshooting:**
        - Ensure `g++` is installed and available in your system's PATH.
        - Use valid API keys for translation models.
        - Check your Python code syntax before translating.
    - **Need More Help?** [Contact Support](https://t.me/SWE_ME)
    """)

# Display the Recent Activity page
def recent_activity_page():
    st.markdown("# 🕒 Recent Activity")
    if "recent_activities" in st.session_state and st.session_state["recent_activities"]:
        for activity in st.session_state["recent_activities"]:
            st.markdown(f"- {activity}")
    else:
        st.markdown("No recent activity yet.")

# Initialize the app
def main():
    # Sidebar with navigation and additional features
    with st.sidebar:
        st.markdown("## 🐍 **ConvertPy**")
        st.markdown("### Convert Python to C++ executables with ease.")
        st.markdown("---")

        choice = option_menu(
            menu_title="Navigation",
            options=["Home", "Convert", "Help", "Recent Activity"],
            icons=["house", "code-slash", "question-circle", "clock"],
            menu_icon="menu-button",
            default_index=0,
        )

    # Display pages based on user choice
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
