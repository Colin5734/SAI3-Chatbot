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
    Öffne ein Terminal oder die Anaconda Powershell und klone dieses Repository:

    ```bash
    git clone <URL_DEINES_GITHUB_REPOS>
    ```

    Navigiere anschliessend in das geklonte Verzeichnis:

    ```bash
    cd SAI3-Chatbot
    ```

2.  **Daten vorbereiten:**

    - Platziere deine Textdatei mit der Wissensbasis in diesem Projektordner.
    - Stelle sicher, dass die Datei den Namen `deine_daten.txt` hat oder passe den `DATA_PATH` Wert oben in der `chatbot.py` Datei entsprechend an.
    - Die Datei sollte im `UTF-8` Encoding vorliegen.

3.  **Conda-Umgebung erstellen und aktivieren:**
    Erstelle eine dedizierte Conda-Umgebung für dieses Projekt (wir nennen sie `sai3_env`) und aktiviere sie:

    ```bash
    # Erstellt die Umgebung mit Python 3.10 (andere Versionen wie 3.9 oder 3.11 gehen oft auch)
    conda create -n sai3_env python=3.10 -y

    # Aktiviert die Umgebung (wichtig für die nächsten Schritte!)
    conda activate sai3_env
    ```

    Dein Terminal-Prompt sollte nun `(sai3_env)` am Anfang anzeigen.

4.  **Abhängigkeiten installieren:**
    Installiere alle notwendigen Python-Pakete mit pip in der **aktiven** `sai3_env` Umgebung:

    ```bash
    pip install langchain langchain-community faiss-cpu sentence-transformers
    ```

5.  **(Für VS Code Nutzer): Python Interpreter auswählen**
    - Öffne den Projektordner in VS Code (`File` -> `Open Folder...`).
    - Öffne die `chatbot.py` Datei.
    - Drücke `Strg+Shift+P` und suche nach `Python: Select Interpreter`.
    - Wähle die Conda-Umgebung `sai3_env` aus der Liste aus. Die Fehler bei den Importen sollten verschwinden.

## Chatbot ausführen

1.  **Ollama Dienst starten:**

    - **Stelle sicher, dass der Ollama Dienst im Hintergrund läuft!**
    - Normalerweise startet er nach der Installation automatisch. Wenn nicht, musst du ihn manuell starten (z.B. über die Ollama App) oder in einem _separaten_ Terminal `ollama serve` ausführen (dieses Terminal muss dann offen bleiben).

2.  **Python-Skript starten:**

    - Öffne die Anaconda Powershell Prompt.
    - Navigiere in den Projektordner (`cd Pfad/zu/SAI3-Chatbot`).
    - **Aktiviere die Conda-Umgebung:** `conda activate sai3_env`
    - Führe das Skript aus:
      ```bash
      python chatbot.py
      ```

3.  **Interagieren:**
    - Beim ersten Start wird der Vektorindex erstellt (dauert einige Minuten). Bei späteren Starts wird der Index geladen (schnell).
    - Wenn "Chatbot ist bereit!" erscheint, kannst du deine Fragen in die Konsole eingeben und mit Enter bestätigen.
    - Tippe `quit` oder `exit`, um den Chatbot zu beenden.

## Wissensbasis aktualisieren

Wenn sich der Inhalt deiner `deine_daten.txt` ändert, musst du den FAISS Vektorindex neu erstellen lassen, damit der Chatbot die neuen Informationen nutzen kann. Lösche dazu einfach den Ordner `faiss_index_gemma_local` im Projektverzeichnis. Beim nächsten Start von `python chatbot.py` wird der Index dann automatisch neu aus der aktuellen `deine_daten.txt` aufgebaut.
