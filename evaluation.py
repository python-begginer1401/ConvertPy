import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import requests
import psutil
import subprocess
from datasets import load_dataset
from Levenshtein import ratio
from typing import Dict, Tuple, List
import re
import threading
import google.generativeai as genai
import os
import platform
from typing import Optional
from datetime import datetime

# ================== CONFIGURATION ==================

CURRENT_USER = "python-begginer1401"

# Set page configuration
st.set_page_config(
    page_title="ConvertPy",
    page_icon="🐍",
    layout="wide"
)

# URL for self-pinging
APP_URL = "https://convertpy2025.streamlit.app/"

def keep_awake(url):
    while True:
        try:
            requests.get(url)
        except Exception:
            pass
        time.sleep(600)

# Start keep-alive thread
threading.Thread(target=keep_awake, args=(APP_URL,), daemon=True).start()

# ================== IMPROVED FEW-SHOT EXAMPLES ==================
FEW_SHOT_EXAMPLES = [
    {
        "py": """def add(a: int, b: int) -> int:
    return a + b""",
        "cpp": """#include <iostream>

int add(int a, int b) {
    return a + b;
}"""
    },
    {
        "py": """def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n-1)""",
        "cpp": """#include <iostream>

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n-1);
}"""
    },
    {
        "py": """class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def area(self) -> float:
        return self.width * self.height""",
        "cpp": """#include <iostream>

class Rectangle {
private:
    double width;
    double height;
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    
    double area() {
        return width * height;
    }
};"""
    }
]

def build_translation_prompt(code: str) -> str:
    """Build a prompt for code translation with examples and clear instructions."""
    
    base_prompt = """Translate this Python code to C++ with exact precision.
Key requirements:
1. Keep exact same logic and functionality
2. Use proper C++ syntax and headers
3. Maintain variable names and structure
4. Handle memory management correctly
5. Include all necessary type declarations
6. Preserve class and function signatures

Here are some examples:"""

    # Add few-shot examples
    for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
        base_prompt += f"\n\nExample {i}:\nPython:\n```python\n{example['py']}\n```\n"
        base_prompt += f"C++:\n```cpp\n{example['cpp']}\n```"

    # Add the actual code to translate
    base_prompt += f"\n\nNow translate this Python code to C++:\n```python\n{code}\n```\n"
    base_prompt += "\nProvide ONLY the C++ code without any explanations."

    return base_prompt

# ================== API RATE LIMITING ==================
# Track last request time for rate limiting
last_request_time = {
    "Gemini": 0,
    "Mistral-small": 0,
    "OpenRouter": 0
}

# Configure rate limits (requests per minute)
RATE_LIMITS = {
    "Gemini": 60,  # 60 requests per minute
    "Mistral-small": 30,  # 30 requests per minute
    "OpenRouter": 10  # 10 requests per minute (more conservative)
}

def rate_limited_request(api_type):
    """Rate limiting decorator to prevent hitting API limits"""
    current_time = time.time()
    time_since_last = current_time - last_request_time[api_type]
    
    # Calculate minimum time between requests (in seconds)
    min_interval = 60.0 / RATE_LIMITS[api_type]
    
    # If we need to wait, then wait
    if time_since_last < min_interval:
        wait_time = min_interval - time_since_last
        time.sleep(wait_time)
    
    # Update the last request time
    last_request_time[api_type] = time.time()

# ================== TRANSLATION FUNCTIONS ==================
def gemini_translate(prompt: str, api_key: str) -> str:
    rate_limited_request("Gemini")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    
    # Add retry logic for potential transient errors
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue
            raise e

def mistral_translate(prompt: str, api_key: str) -> str:
    rate_limited_request("Mistral-small")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-small",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert C++ programmer. Convert Python code to C++ exactly."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1500,
        "top_p": 0.95
    }
    
    # Add retry logic
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content']
            raise Exception(f"Mistral API error: {response_data}")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue
            raise e

def openrouter_translate(prompt: str, api_key: str, model_id: str) -> str:
    rate_limited_request("OpenRouter")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://localhost:8501",
        "X-Title": "ConvertPy"
    }
    
    messages = [
        {
            "role": "system",
            "content": "You are an expert C++ programmer. Convert Python code to C++ exactly."
        },
        {
            "role": "user",
            "content": f"Convert this Python code to C++:\n\n{prompt}"
        }
    ]
    
    # Add retry logic with longer delays for OpenRouter
    max_retries = 4
    retry_delay = 3  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": model_id,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 1500,
                    "top_p": 0.95,
                    "response_format": {"type": "text"}
                },
                timeout=30
            )
            
            # Special handling for rate limits
            if response.status_code == 429:
                # Wait longer for rate limit errors
                wait_time = (retry_delay * (2 ** attempt))  # Exponential backoff
                st.warning(f"OpenRouter rate limit hit. Waiting {wait_time} seconds before retry.")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                if "429" in str(e):
                    wait_time = 10 * (attempt + 1)  # Wait longer for rate limits
                time.sleep(wait_time)
                continue
            # After all retries, propagate the error
            raise Exception(f"OpenRouter API error after {max_retries} retries: {str(e)}")

# ================== MODEL CONFIG ==================
MODELS = {
    "Gemini": {
        "translator": gemini_translate,
        "key": "Gemini",
        "color": "#4285F4"
    },
    "Mistral-small": {
        "translator": mistral_translate,
        "key": "Mistral-small",
        "color": "#FF6D00"
    },
    "DeepSeek": {
        "translator": lambda p, k: openrouter_translate(p, k, "deepseek/deepseek-prover-v2:free"),
        "key": "OpenRouter",
        "color": "#00BFA5"
    },
    "Llama": {
        "translator": lambda p, k: openrouter_translate(p, k, "meta-llama/llama-3.3-8b-instruct:free"),
        "key": "OpenRouter",
        "color": "#FF5252"
    }
}

# ================== EVALUATION FUNCTIONS ==================
def calculate_structure_preservation(py_code: str, cpp_code: str) -> float:
    """
    Calculates structure preservation score by comparing key structural elements
    between Python and C++ code.
    """
    if not py_code or not cpp_code:
        return 0.0
    
    # Extract key structural elements
    py_elements = {
        'functions': len(re.findall(r'def\s+(\w+)\s*\(', py_code)),
        'classes': len(re.findall(r'class\s+(\w+)', py_code)),
        'loops': len(re.findall(r'for\s+|while\s+', py_code)),
        'conditionals': len(re.findall(r'if\s+|elif\s+|else:', py_code)),
        'returns': len(re.findall(r'return\s+', py_code)),
        'variables': len(set(re.findall(r'(\w+)\s*=', py_code)))
    }
    
    cpp_elements = {
        'functions': len(re.findall(r'(?:int|void|float|double|bool|auto|string)\s+(\w+)\s*\(', cpp_code)),
        'classes': len(re.findall(r'class\s+(\w+)', cpp_code)),
        'loops': len(re.findall(r'for\s*\(|while\s*\(', cpp_code)),
        'conditionals': len(re.findall(r'if\s*\(|else\s+if|else\s*{', cpp_code)),
        'returns': len(re.findall(r'return\s+', cpp_code)),
        'variables': len(set(re.findall(r'(?:int|float|double|bool|auto|string)\s+(\w+)', cpp_code)))
    }
    
    # Calculate preservation scores for each element
    scores = []
    for key in py_elements:
        py_count = py_elements[key]
        cpp_count = cpp_elements[key]
        
        if py_count > 0:
            # Ratio of found elements, capped at 100%
            preservation = min(100, (cpp_count / py_count) * 100)
            scores.append(preservation)
    
    # If we have scores, return their average, otherwise 0
    if scores:
        return sum(scores) / len(scores)
    else:
        return 0.0

def calculate_translation_completeness(py_code: str, cpp_code: str) -> float:
    """
    Calculates what percentage of the Python code was attempted to be translated
    """
    if not py_code or not cpp_code:
        return 0.0
    
    # Get all identifiers from Python code (variable names, function names, etc.)
    py_identifiers = set(re.findall(r'\b([a-zA-Z_]\w*)\b', py_code))
    
    # Remove Python keywords
    py_keywords = {"and", "as", "assert", "break", "class", "continue", "def", 
                  "del", "elif", "else", "except", "False", "finally", "for", 
                  "from", "global", "if", "import", "in", "is", "lambda", "None", 
                  "nonlocal", "not", "or", "pass", "raise", "return", "True", 
                  "try", "while", "with", "yield"}
    py_identifiers = py_identifiers - py_keywords
    
    # Count identifiers that appear in C++ code
    cpp_lower = cpp_code.lower()
    found_identifiers = sum(1 for id in py_identifiers if id.lower() in cpp_lower)
    
    # Calculate percentage of identifiers preserved
    if len(py_identifiers) > 0:
        completeness = (found_identifiers / len(py_identifiers)) * 100
    else:
        # If no identifiers found, check if there's any code at all
        completeness = 100 if len(cpp_code) > 20 else 0
    
    # Adjust score based on relative code size
    py_lines = len(py_code.strip().splitlines())
    cpp_lines = len(cpp_code.strip().splitlines())
    
    # A good C++ translation is typically 1-3x the number of Python lines
    # Penalize if too short, but don't penalize for being longer
    if py_lines > 0:
        size_ratio = cpp_lines / py_lines
        if size_ratio < 0.5:  # Too short
            completeness *= size_ratio * 2  # Scale down proportionally
    
    return min(100, completeness)

def has_valid_cpp_syntax(cpp_code: str) -> bool:
    """
    Check if the code has basic valid C++ syntax markers.
    Not a full compiler check, just basic indicators.
    """
    # Check for common C++ syntax elements
    has_semicolons = ";" in cpp_code
    has_brackets = "{" in cpp_code and "}" in cpp_code
    has_headers = "#include" in cpp_code
    has_valid_types = bool(re.search(r'\b(int|void|double|float|bool|string|auto)\b', cpp_code))
    
    # A more advanced check could be to count the number of open/close brackets
    balanced_brackets = cpp_code.count("{") == cpp_code.count("}")
    balanced_parentheses = cpp_code.count("(") == cpp_code.count(")")
    
    # Syntax should have most of these elements
    syntax_score = sum([has_semicolons, has_brackets, has_headers, has_valid_types, 
                      balanced_brackets, balanced_parentheses])
    
    return syntax_score >= 4  # At least 4 out of 6 syntax elements should be present

def clean_code(out: str) -> str:
    """Extract clean C++ code from model response"""
    # Extract code from markdown blocks
    cpp_pattern = r"```(?:cpp|c\+\+)(.*?)```"
    matches = re.findall(cpp_pattern, out, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # If no markdown blocks, try finding C++ code based on common patterns
    lines = []
    in_code = False
    for line in out.split('\n'):
        line = line.strip()
        
        # Look for C++ indicators
        if not in_code and ('#include' in line or 'int main' in line or 'class ' in line):
            in_code = True
            
        if in_code and not line.startswith(('Here', 'Note:', 'This is', 'In this')):
            lines.append(line)
    
    if lines:
        return '\n'.join(lines).strip()
    
    # Fallback: clean the entire output removing non-code parts
    lines = []
    for line in out.split('\n'):
        line = line.strip()
        if line and not line.startswith(('```', '//', '/*', '*', 'Note:', 'Here', 'This is')):
            lines.append(line)
    
    return '\n'.join(lines).strip()

def measure_size(code: str) -> int:
    """Measure size of code in lines"""
    return len(code.strip().splitlines())

def evaluate_sample(model: str, code_py: str, code_cpp: str, key: str) -> Tuple[float, float, int, int, int, bool]:
    if not key:
        return 0.0, 0.0, 0, measure_size(code_py), 0, False
    
    prompt = build_translation_prompt(code_py)
    start = time.time()

    try:
        out = MODELS[model]["translator"](prompt, key)
        if not out:
            print(f"Empty response from {model}")
            return 0.0, time.time() - start, 0, measure_size(code_py), 0, False
            
        gen = clean_code(out)
        if not gen:
            print(f"No code extracted from {model} response")
            return 0.0, time.time() - start, 0, measure_size(code_py), 0, False
            
        # Calculate new metrics
        structure_score = calculate_structure_preservation(code_py, gen)
        completeness_score = calculate_translation_completeness(code_py, gen)
        
        # Get output length in lines
        output_length = len(gen.splitlines())
        
        # Check if C++ syntax looks valid
        valid_syntax = has_valid_cpp_syntax(gen)
        
        print(f"{model} structure preservation: {structure_score:.2f}% for sample")
        
    except Exception as e:
        print(f"Error evaluating {model}: {str(e)}")
        return 0.0, time.time() - start, 0, measure_size(code_py), 0, False

    elapsed = time.time() - start
    size = measure_size(code_py)

    return structure_score, elapsed, output_length, size, completeness_score, valid_syntax

# ================== BATCH PROCESSING ==================
def process_batch(batch_samples: List[dict], models: List[str], keys: Dict[str, str], results: Dict, sizes: Dict):
    """Process a batch of samples across all models"""
    for row in batch_samples:
        for m in models:
            api_key = keys[MODELS[m]["key"]]
            if not api_key:
                continue
                
            try:
                structure, t, output_len, il, completeness, valid_syntax = evaluate_sample(
                    m, 
                    row["code"],
                    row["code"],  # Using same code as reference for now
                    api_key
                )
                
                results[m]["structure"].append(structure)
                results[m]["time"].append(t)
                results[m]["output_length"].append(output_len)
                results[m]["completeness"].append(completeness)
                results[m]["valid_syntax"].append(valid_syntax)
                sizes[m].append(il)
            except Exception as e:
                print(f"Error processing sample for {m}: {str(e)}")
    
    return results, sizes

# ================== PLOTTING ==================
def create_plots(results: Dict, sizes: Dict, n: int):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("ConvertPy Model Evaluation", fontsize=18, y=1.02)

    # Structure Preservation Score (replacing Accuracy)
    ax = axes[0, 0]
    structure_scores = [np.mean(results[m]["structure"]) for m in results]
    bars = ax.bar(results.keys(), structure_scores, color=[MODELS[m]["color"] for m in results])
    ax.set_ylim(0, 100)
    ax.set_title("Structure Preservation (%)")
    ax.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Target (70%)')
    for b, h in zip(bars, structure_scores):
        ax.text(b.get_x()+b.get_width()/2, h+1, f"{h:.1f}%", ha="center")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    # Time vs Input Size with jitter
    ax = axes[0, 1]
    for m in results:
        x_values = np.array(sizes[m])
        jitter = np.random.normal(0, 0.1, len(x_values))
        ax.scatter(x_values + jitter, results[m]["time"],
                   label=m, color=MODELS[m]["color"],
                   s=60, alpha=0.7)
    ax.set_xlabel("Lines of Code")
    ax.set_ylabel("Time (s)")
    ax.set_title("Translation Time vs Input Size")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    # Translation Completeness (replacing Success Rate)
    ax = axes[1, 0]
    completeness = [np.mean(results[m]["completeness"]) for m in results]
    bars = ax.bar(results.keys(), completeness, color=[MODELS[m]["color"] for m in results])
    ax.set_ylim(0, 100)
    ax.set_title("Translation Completeness (%)")
    for b, h in zip(bars, completeness):
        ax.text(b.get_x()+b.get_width()/2, h+1, f"{h:.1f}%", ha="center")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Average Output Length (lines)
    ax = axes[1, 1]
    output_lengths = [np.mean(results[m]["output_length"]) for m in results]
    bars = ax.bar(results.keys(), output_lengths, color=[MODELS[m]["color"] for m in results])
    ax.set_title("Average Output Length (Lines)")
    for b, h in zip(bars, output_lengths):
        ax.text(b.get_x()+b.get_width()/2, h+0.5, f"{h:.1f}", ha="center")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig

# ================== MAIN APP ==================
def main():
    st.title("🐍 ConvertPy: Model Evaluation")
    st.markdown(f"Current Date and Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Load dataset
        ds = load_dataset("codeparrot/xlcost-text-to-code", "Python-program-level", split="train")
        df = pd.DataFrame(ds)
        
        if len(df) == 0:
            st.error("Failed to load dataset.")
            return

        max_samp = min(100, len(df))
        samp = st.sidebar.slider("Samples to evaluate", 5, max_samp, min(20, max_samp))
        models = st.sidebar.multiselect("Models", list(MODELS.keys()), default=list(MODELS.keys()))
        
        # Batch size configuration
        batch_size = st.sidebar.slider("Batch size", 1, 5, 2, 
                                      help="Number of samples to process at once. Lower values help avoid rate limits.")
        
        st.sidebar.subheader("API Keys")
        keys = {
            "Gemini": st.sidebar.text_input("Gemini API Key", type="password"),
            "Mistral-small": st.sidebar.text_input("Mistral API Key", type="password"),
            "OpenRouter": st.sidebar.text_input("OpenRouter API Key", type="password"),
        }

        if st.button("🚀 Run Enhanced Evaluation"):
            if not any(keys.values()):
                st.error("Please provide at least one API key.")
                return
                
            results = {m: {
                "structure": [],  # Replaced accuracy with structure preservation
                "time": [], 
                "output_length": [],
                "completeness": [],  # Replaced success with completeness
                "valid_syntax": []  # Added syntax validation
            } for m in models}
            sizes = {m: [] for m in models}
            
            # Filter dataset
            df['code_length'] = df['code'].apply(lambda x: len(x.split('\n')))
            
            # Try multiple filtering approaches
            filtered_df = df[(df['code_length'] >= 1) & (df['code_length'] <= 50)]  # Wider range
            
            # Check if we have any samples after filtering
            if len(filtered_df) == 0:
                st.warning("No samples match the length criteria. Using the full dataset sorted by length.")
                df_samp = df.sort_values(by='code_length').head(samp).reset_index(drop=True)
            elif len(filtered_df) < samp:
                st.warning(f"Only {len(filtered_df)} samples meet the length criteria (need {samp}). Using all available samples.")
                df_samp = filtered_df.reset_index(drop=True)
            else:
                df_samp = filtered_df.sample(min(samp, len(filtered_df)), random_state=42).reset_index(drop=True)
            
            progress = st.progress(0)
            status = st.empty()
            st_metrics = st.empty()
            batch_status = st.empty()

            try:
                # Split samples into batches
                sample_list = df_samp.to_dict('records')
                num_batches = max(1, len(sample_list) // batch_size)
                batches = [sample_list[i:i + batch_size] for i in range(0, len(sample_list), batch_size)]
                
                for i, batch in enumerate(batches):
                    batch_status.text(f"Processing batch {i+1}/{len(batches)}...")
                    
                    # Process the batch
                    for sample_idx, row in enumerate(batch):
                        status.text(f"Sample {i*batch_size + sample_idx + 1}/{len(df_samp)}")
                        current_metrics = []
                        
                        for m in models:
                            api_key = keys[MODELS[m]["key"]]
                            if not api_key:
                                continue
                                
                            structure, t, output_len, il, completeness, valid_syntax = evaluate_sample(
                                m, 
                                row["code"],
                                row["code"],  # Using same code as reference for now
                                api_key
                            )
                            
                            results[m]["structure"].append(structure)
                            results[m]["time"].append(t)
                            results[m]["output_length"].append(output_len)
                            results[m]["completeness"].append(completeness)
                            results[m]["valid_syntax"].append(valid_syntax)
                            sizes[m].append(il)
                            
                            current_metrics.append(f"{m}: {structure:.1f}%")
                        
                        st_metrics.text(" | ".join(current_metrics))
                    
                    # Update progress after each batch
                    progress.progress(min(1.0, (i+1)/len(batches)))
                    
                    # Add a delay between batches to avoid rate limits
                    if i < len(batches) - 1:
                        batch_delay = 5  # seconds between batches
                        batch_status.text(f"Completed batch {i+1}/{len(batches)}. Waiting {batch_delay}s before next batch...")
                        time.sleep(batch_delay)

                st.success("✅ Evaluation complete!")
                st.pyplot(create_plots(results, sizes, len(df_samp)))

                # Summary Statistics
                summary = []
                for m in models:
                    if results[m]["structure"]:
                        avg_structure = np.mean(results[m]["structure"])
                        avg_completeness = np.mean(results[m]["completeness"])
                        syntax_rate = sum(results[m]["valid_syntax"]) / len(results[m]["valid_syntax"]) * 100
                        
                        summary.append({
                            "Model": m,
                            "Structure Preservation (%)": f"{avg_structure:.1f}" + (" ✅" if avg_structure >= 70 else " ⚠️"),
                            "Translation Completeness (%)": f"{avg_completeness:.1f}%",
                            "Valid Syntax Rate (%)": f"{syntax_rate:.1f}%",
                            "Avg Time (s)": f"{np.mean(results[m]['time']):.2f}",
                            "Avg Output (Lines)": f"{np.mean(results[m]['output_length']):.1f}",
                        })
                
                st.subheader("📊 Summary Statistics")
                if summary:
                    st.dataframe(pd.DataFrame(summary), use_container_width=True)
                else:
                    st.error("No results to display. Please check API keys and try again.")

            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")
                
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    st.sidebar.markdown(f"_Created by {CURRENT_USER}_")
    main()