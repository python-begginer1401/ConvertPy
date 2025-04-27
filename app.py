import streamlit as st
import google.generativeai as genai
import subprocess
import os
import platform
from io import BytesIO
import textwrap
import requests
import time

while True:
    requests.get("https://convertpy.streamlit.app/")
    time.sleep(300)  # Ping every 5 minutes


# Sidebar for API Key input and tab selection
with st.sidebar:
    st.sidebar.title("Navigation")
    tabs = st.sidebar.radio("Select an option", ["🏠 Home", "📝 Convert Python to Executable"])
    api_key = st.text_input("Google API Key", key="geminikey", type="password")

# Initialize session states
if "translated_code" not in st.session_state:
    st.session_state["translated_code"] = ""
if "compile_clicked" not in st.session_state:
    st.session_state["compile_clicked"] = False

def get_os_info():
    """Returns (os_name, executable_extension) tuple"""
    system = platform.system().lower()
    if system == "darwin":
        return "macOS", ""
    elif system == "windows":
        return "Windows", ".exe"
    elif system == "linux":
        return "Linux", ""
    else:
        return system.capitalize(), ""

def compile_cpp_code(cpp_code, file_name="program"):
    """Cross-platform compilation function"""
    cpp_file_path = f"{file_name}.cpp"
    os_name, exe_ext = get_os_info()
    exe_name = f"{file_name}{exe_ext}"
    
    # Save the C++ code to a file
    with open(cpp_file_path, "w") as cpp_file:
        cpp_file.write(cpp_code)
    
    # Platform-specific compilation flags
    compile_command = ["g++", cpp_file_path, "-o", exe_name]
    if os_name != "Windows":
        compile_command.extend(["-std=c++11", "-pthread"])  # Common Unix flags
    
    try:
        result = subprocess.run(compile_command, capture_output=True, text=True)
        if result.returncode == 0:
            st.success(f"Compilation successful for {os_name}!")
            return exe_name
        else:
            st.error(f"Compilation error:\n{result.stderr}")
            return None
    except FileNotFoundError:
        st.error(f"Compiler not found. Please install g++ on your {os_name} system.")
        if os_name == "macOS":
            st.info("On macOS, install Xcode Command Line Tools with: xcode-select --install")
        elif os_name == "Linux":
            st.info("On Linux, install with: sudo apt-get install build-essential")
        return None
    except Exception as e:
        st.error(f"An error occurred during compilation: {e}")
        return None
    finally:
        # Clean up the C++ file
        if os.path.exists(cpp_file_path):
            os.remove(cpp_file_path)

# Main Page Tab
if tabs == "🏠 Home":
    st.title("🐍 ConvertPy (Cross-Platform)")
    os_name, _ = get_os_info()
    st.write(f"""
        Welcome to ConvertPy! 
        
        This tool converts Python code to C++ and compiles it for your operating system.
        Detected OS: **{os_name}**
        
        Current supported platforms: Windows, Linux, and macOS.
        
        Select the conversion tab from the sidebar to get started!
    """)
    st.info("Note: You'll need g++ compiler installed on your system.")

# Convert Python to Executable Tab
elif tabs == "📝 Convert Python to Executable":
    st.title("📝 Convert Python to Executable")
    os_name, exe_ext = get_os_info()
    st.write(f"Detected OS: **{os_name}**")
    
    text_input = st.text_area("Enter your Python code 🐍", height=200, 
                            placeholder="Write your Python code here...")

    if st.button("Translate Code to C++") and text_input:
        if not api_key:
            st.error("Please enter your Google API Key in the sidebar")
            st.stop()
            
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-pro-latest')
            
            prompt = f"""Convert the following Python code to standard C++11 code:
            {text_input}
            
            Requirements:
            1. Use only standard C++11 features
            2. Make it portable across Windows, Linux and macOS
            3. Include all necessary headers
            4. Return only the pure C++ code with no additional explanations
            5. Ensure the code is properly indented
            6. Include input/output handling if needed
            7. Use modern C++ practices
            8. Avoid platform-specific APIs
            """
            
            response = model.generate_content(prompt)
            
            # Extract the generated C++ code
            if response and hasattr(response, 'text'):
                cpp_code = response.text
            elif response.candidates:
                cpp_code = response.candidates[0].content.parts[0].text
            else:
                st.error("No valid response from the AI model")
                st.stop()
                
            # Clean up the response
            cpp_code = cpp_code.strip()
            if cpp_code.startswith("```cpp"):
                cpp_code = cpp_code[6:]
            if cpp_code.endswith("```"):
                cpp_code = cpp_code[:-3]
            cpp_code = cpp_code.strip()
            
            st.session_state["translated_code"] = cpp_code
            st.session_state["compile_clicked"] = False

            st.write("### Translated C++ Code")
            st.code(st.session_state["translated_code"], language='cpp')

        except Exception as e:
            st.error(f"Translation failed: {str(e)}")
            st.error("Please ensure you're using the correct model name and API version")

    if st.session_state.get("translated_code"):
        st.write("### Compilation Options")
        if st.button("Compile for My System"):
            st.session_state["compile_clicked"] = True

    if st.session_state.get("compile_clicked"):
        with st.spinner("Compiling..."):
            exe_file = compile_cpp_code(st.session_state["translated_code"])
            
            if exe_file:
                try:
                    with open(exe_file, "rb") as f:
                        exe_bytes = f.read()
                    
                    st.download_button(
                        label=f"Download Executable for {os_name}",
                        data=exe_bytes,
                        file_name=f"program{exe_ext}",
                        mime="application/octet-stream"
                    )
                except Exception as e:
                    st.error(f"Failed to prepare download: {e}")
                finally:
                    if os.path.exists(exe_file):
                        os.remove(exe_file)
