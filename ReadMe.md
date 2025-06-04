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

3.  **Prepare Data:**

    - Ensure your text file, serving as the knowledge base, is located in this project folder.
    - Make sure the file is named `Mahabharata_Gita_Light_Edition.txt`.
    - The file should be in `UTF-8` encoding.

4.  **Create and Activate Conda Environment:**
    Create a dedicated Conda environment for this project (we'll call it `sai3_env`) and activate it:

    ```bash
    # Creates the environment with Python 3.10 (other versions like 3.9 or 3.11 often work too)
    conda create -n sai3_env python=3.10 -y

    # Activates the environment (important for the next steps!)
    conda activate sai3_env
    ```

    Your terminal prompt should now show `(sai3_env)` at the beginning.
    *(Note: On macOS/Linux, this happens in the standard terminal; on Windows, in the Anaconda Powershell Prompt or CMD, after Anaconda has been added to the PATH or `conda init` has been run.)*

5.  **Install Dependencies:**
    Install all necessary Python packages using pip in the **active** `sai3_env` environment:

    ```bash
    pip install langchain langchain-community faiss-cpu sentence-transformers gradio
    ```

6.  **(For VS Code Users): Select Python Interpreter**
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

---

# SAI3 RAG Chatbot mit Ollama

Dieses Projekt implementiert einen Retrieval-Augmented Generation (RAG) Chatbot in Python. Der Chatbot nutzt eine lokale Textdatei als Wissensbasis und beantwortet Fragen mithilfe des lokal laufenden Sprachmodells über Ollama.

**Kerntechnologien:**

- Python
- LangChain (Framework für LLM-Anwendungen)
- Ollama (Lokales Ausführen von LLMs)
- FAISS (Vektordatenbank für Ähnlichkeitssuche)
- Sentence Transformers (Erstellung von Text-Embeddings)

## Voraussetzungen

Bevor du beginnst, stelle sicher, dass folgende Software auf deinem System installiert ist:

1.  **Git:** Zur Versionsverwaltung und zum Klonen des Repositories ([git-scm.com](https://git-scm.com/)).
2.  **Anaconda:** Zur Verwaltung der Python-Umgebung ([anaconda.com/download](https://anaconda.com/download)).
3.  **Ollama:** Zum lokalen Ausführen des Sprachmodells ([ollama.ai](https://ollama.ai/)).
4.  **Ein Ollama LLM:** Lade das spezifische Modell herunter, das dieses Projekt verwendet:
    ```bash
    # Öffne ein Terminal oder eine Powershell
    ollama pull mistral:7b-instruct-v0.2-q4_K_M
    ```
    Du kannst mit `ollama list` überprüfen, welche Modelle installiert sind.

## Setup-Anleitung

1.  **Repository klonen:**  
    Öffne ein **Terminal** (auf macOS/Linux) oder die **Anaconda Powershell Prompt / CMD** (auf Windows) und klone dieses Repository:
    
    ```bash
    git clone https://github.com/Colin5734/SAI3-Chatbot.git
    ```

    Navigiere anschließend in das geklonte Verzeichnis:

    ```bash
    cd SAI3-Chatbot
    ```

3.  **Daten vorbereiten:**

    - Überprüfe ob deine Textdatei mit der Wissensbasis in diesem Projektordner liegt.
    - Stelle sicher, dass die Datei den Namen `Mahabharata_Gita_Light_Edition.txt` hat.
    - Die Datei sollte im `UTF-8` Encoding vorliegen.

4.  **Conda-Umgebung erstellen und aktivieren:**  
    Erstelle eine dedizierte Conda-Umgebung für dieses Projekt (wir nennen sie `sai3_env`) und aktiviere sie:

    ```bash
    # Erstellt die Umgebung mit Python 3.10 (andere Versionen wie 3.9 oder 3.11 gehen oft auch)
    conda create -n sai3_env python=3.10 -y

    # Aktiviert die Umgebung (wichtig für die nächsten Schritte!)
    conda activate sai3_env
    ```

    Dein Terminal-Prompt sollte nun `(sai3_env)` am Anfang anzeigen.  
    *(Hinweis: Unter macOS/Linux geschieht dies im Standard-Terminal, unter Windows in der Anaconda Powershell Prompt oder CMD, nachdem Anaconda zum PATH hinzugefügt wurde oder `conda init` ausgeführt wurde.)*

5.  **Abhängigkeiten installieren:**  
    Installiere alle notwendigen Python-Pakete mit pip in der **aktiven** `sai3_env` Umgebung:

    ```bash
    pip install langchain langchain-community faiss-cpu sentence-transformers gradio
    ```

6.  **(Für VS Code Nutzer): Python Interpreter auswählen**
    - Öffne den Projektordner in VS Code (`File` → `Open Folder...`).
    - Öffne die `chatbot.py` Datei.
    - Drücke `Strg+Shift+P` und suche nach `Python: Select Interpreter`.
    - Wähle die Conda-Umgebung `sai3_env` aus der Liste aus. Die Fehler bei den Importen sollten verschwinden.

## Chatbot ausführen

1.  **Ollama Dienst starten:**

    - **Stelle sicher, dass der Ollama Dienst im Hintergrund läuft!**
    - Normalerweise startet er nach der Installation automatisch. Wenn nicht, musst du ihn manuell starten (z.B. über die Ollama App) oder in einem _separaten_ Terminal `ollama serve` ausführen (dieses Terminal muss dann offen bleiben).

2.  **Python-Skript starten:**

    - Öffne ein **Terminal** (macOS/Linux) oder die **Anaconda Powershell Prompt / CMD** (Windows).
    - Navigiere in den Projektordner (`cd Pfad/zu/SAI3-Chatbot`).
    - **Aktiviere die Conda-Umgebung:** `conda activate sai3_env`
    - Führe das Skript aus:
      ```bash
      python ui.py
      ```
      *(Hinweis: Je nach Python-Installation muss eventuell `python3 ui.py` verwendet werden, besonders auf macOS/Linux, wenn `python` auf eine ältere Systemversion zeigt.)*

3.  **Interagieren:**
    - Beim ersten Start wird der Vektorindex erstellt (dauert einige Minuten). Bei späteren Starts wird der Index geladen (schnell).
    - Danach öffnet sich automatisch dein Standard-Browser mit der Adresse http://localhost:7860.
    - Gib deine Frage in das Eingabefeld („Type your question here…“) ein und drücke Enter oder klicke Submit.
    - Die Frage erscheint links (mit User-Avatar), die Antwort rechts (mit Bot-Avatar).
    - Nach Absenden wird das Eingabefeld automatisch geleert.
    - Zum Beenden: Drücke Ctrl + C im Terminal, wo python ui.py läuft.

## Wissensbasis aktualisieren

Wenn sich der Inhalt deiner `Mahabharata_Gita_Light_Edition.txt` ändert, musst du den FAISS Vektorindex neu erstellen lassen, damit der Chatbot die neuen Informationen nutzen kann. Lösche dazu einfach den Ordner `faiss_index_gemma_local` im Projektverzeichnis. Beim nächsten Start von `python chatbot.py` wird der Index dann automatisch neu aus der aktuellen `Mahabharata_Gita_Light_Edition.txt` aufgebaut.
