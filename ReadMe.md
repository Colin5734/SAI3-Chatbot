## Group A
Luca Lüdi, Colin Marti, Hristian Petrov, Cyril Schlup, Sriprakatheeswaran Thiraviyachelvam

# SAI3 RAG Chatbot with Ollama

This project implements a Retrieval-Augmented Generation (RAG) chatbot in Python. The chatbot uses a local text file as its knowledge base and answers questions using a locally running language model via Ollama.

**Core Technologies:**

- Python
- LangChain (Framework for LLM applications)
- Ollama (Running LLMs locally)
- FAISS (Vector database for similarity search)
- Sentence Transformers (Creation of text embeddings)

## Prerequisites

Before you begin, ensure the following software is installed on your system:

1.  **Git:** For version control and cloning the repository ([git-scm.com](https://git-scm.com/)).
2.  **Anaconda:** For managing the Python environment ([anaconda.com/download](https://anaconda.com/download)).
3.  **Ollama:** For running the language model locally ([ollama.ai](https://ollama.ai/)).
4.  **An Ollama LLM:** Download the specific model used by this project:
    ```bash
    # Open a terminal or PowerShell
    ollama pull mistral:7b-instruct-v0.2-q4_K_M
    ```
    You can check which models are installed with `ollama list`.

## Setup Instructions

1.  **Clone Repository:**
    Open a **Terminal** (on macOS/Linux) or the **Anaconda Powershell Prompt / CMD** (on Windows) and clone this repository:

    ```bash
    git clone https://github.com/Colin5734/SAI3-Chatbot.git
    ```

    Then navigate into the cloned directory:

    ```bash
    cd SAI3-Chatbot
    ```

2.  **Prepare Data:**

    - Ensure your text file, serving as the knowledge base, is located in this project folder.
    - Make sure the file is named `Mahabharata_Gita_Light_Edition.txt`.
    - The file should be in `UTF-8` encoding.

3.  **Create and Activate Conda Environment:**
    Create a dedicated Conda environment for this project (we'll call it `sai3_env`) and activate it:

    ```bash
    # Creates the environment with Python 3.10 (other versions like 3.9 or 3.11 often work too)
    conda create -n sai3_env python=3.10 -y

    # Activates the environment (important for the next steps!)
    conda activate sai3_env
    ```

    Your terminal prompt should now show `(sai3_env)` at the beginning.
    *(Note: On macOS/Linux, this happens in the standard terminal; on Windows, in the Anaconda Powershell Prompt or CMD, after Anaconda has been added to the PATH or `conda init` has been run.)*

4.  **Install Dependencies:**
    Install all necessary Python packages using pip in the **active** `sai3_env` environment:
    
    First, install the FAISS library using conda from the pytorch channel

    ```bash
    conda install -c pytorch faiss-cpu -y
    ```

    Then, install the other Python packages using pip

    ```bash
    pip install langchain langchain-community sentence-transformers gradio
    ```

5.  **(Only for VS Code Users if there are import problems within the files): Select Python Interpreter**
    - Open the project folder in VS Code (`File` → `Open Folder...`).
    - Open the `chatbot.py` file.
    - Press `Ctrl+Shift+P` and search for `Python: Select Interpreter`.
    - Select the `sai3_env` Conda environment from the list. The import errors should disappear.

## Run Chatbot

1.  **Start Ollama Service:**

    - **Ensure the Ollama service is running in the background!**
    - It usually starts automatically after installation. If not, you need to start it manually (e.g., via the Ollama app) or run `ollama serve` in a _separate_ terminal (this terminal must then remain open).

2.  **Start Python Script:**

    - Open a **Terminal** (macOS/Linux) or the **Anaconda Powershell Prompt / CMD** (Windows).
    - Navigate to the project folder (`cd path/to/SAI3-Chatbot`).
    - **Activate the Conda environment:** `conda activate sai3_env`
    - Run the script:
      ```bash
      python ui.py
      ```
      *(Note: Depending on your Python installation, you might need to use `python3 ui.py`, especially on macOS/Linux if `python` points to an older system version.)*

3.  **Interact:**
    - On the first run, the vector index will be created (takes a few minutes). On subsequent runs, the index will be loaded (fast).
    - Afterwards, your default browser will automatically open to the address http://localhost:7860.
    - Enter your question in the input field ("Type your question here…") and press Enter or click Submit.
    - The question appears on the left (with a user avatar), the answer on the right (with a bot avatar).
    - After submitting, the input field is automatically cleared.
    - To exit: Press Ctrl + C in the terminal where `python ui.py` is running.

## Update Knowledge Base

If the content of your `Mahabharata_Gita_Light_Edition.txt` changes, you need to have the FAISS vector index rebuilt so that the chatbot can use the new information. To do this, simply delete the `faiss_index_gemma_local` folder in the project directory. The next time you start `python chatbot.py`, the index will then be automatically rebuilt from the current `Mahabharata_Gita_Light_Edition.txt`.
