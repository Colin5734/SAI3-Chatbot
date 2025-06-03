# ui.py

import gradio as gr
import sys
import html    # Zum Escapen von Sonderzeichen
import base64  # Um Bilder in Base64 zu encoden
from pathlib import Path
import webbrowser  # Für automatisches Öffnen im Standard‐Browser

from chatbot import (
    load_or_create_vectorstore,
    create_rag_chain,
    DATA_PATH,
    VECTORSTORE_PATH,
    EMBEDDING_MODEL_NAME,
    OLLAMA_MODEL_NAME,
    INDEX_BATCH_SIZE
)

# --------------------------------------------------------------------------------
# 0. AVATAR-BILDER VORAB IN BASE64 EINLESEN (damit keine 404-Fehler auftreten)

user_avatar_path = Path("images") / "user_avatar.jpg"
bot_avatar_path  = Path("images") / "bot_avatar.jpg"

def encode_image_to_base64(path: Path) -> str:
    """Liest eine Bilddatei ein und gibt sie als Base64-String zurück."""
    if not path.exists():
        print(f"❌ Avatar file not found: {path}")
        return ""
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")

user_avatar_b64 = encode_image_to_base64(user_avatar_path)
bot_avatar_b64  = encode_image_to_base64(bot_avatar_path)

if not user_avatar_b64 or not bot_avatar_b64:
    print("❌ One or both avatar images could not be loaded. Make sure 'images/user_avatar.jpg' "
          "and 'images/bot_avatar.jpg' exist.")
    sys.exit(1)

# --------------------------------------------------------------------------------
# 1. FAISS-Index laden oder neu erstellen (einmalig beim Start)
print("ℹ️ Loading or creating the FAISS index …")
try:
    vectorstore = load_or_create_vectorstore(
        DATA_PATH,
        VECTORSTORE_PATH,
        EMBEDDING_MODEL_NAME,
        INDEX_BATCH_SIZE
    )
except Exception as e:
    print(f"✖️ Error loading/creating vectorstore: {e}")
    sys.exit(1)

# 2. RAG-Chain initialisieren (Ollama + Retriever)
print("ℹ️ Initializing the RAG chain …")
try:
    rag_chain = create_rag_chain(vectorstore, OLLAMA_MODEL_NAME)
except Exception as e:
    print(f"✖️ Error initializing RAG chain: {e}")
    sys.exit(1)


# 3. Callback-Funktion für den Chat
def chat_with_bot(user_input, history):
    """
    user_input: String – the new user question
    history:    List of {"role": ..., "content": ...} dicts or []/None on first call
    """
    if history is None:
        history = []

    # LLM aufrufen
    output = rag_chain.invoke({"query": user_input})
    answer = output["result"]

    # OpenAI-Style-History anreichern
    history.append({"role": "user",      "content": user_input})
    history.append({"role": "assistant", "content": answer})

    # Rückgabe:
    # 1) HTML-Darstellung
    # 2) aktualisierte History
    # 3) leeres String, damit das Textfeld zurückgesetzt wird
    return render_chat_html(history), history, ""


# 4. Hilfsfunktion: Aus History einen HTML-String mit ChatGPT-ähnlichem Stil bauen
def render_chat_html(history):
    html_chunks = []
    for msg in history:
        role    = msg["role"]
        content = html.escape(msg["content"])

        if role == "user":
            avatar_uri = f"data:image/jpeg;base64,{user_avatar_b64}"
            html_chunks.append(f"""
                <div style="
                    display: flex;
                    align-items: center;   /* Avatar zentriert vertikal */
                    margin: 6px 0;
                ">
                  <!-- User-Icon (64×64) -->
                  <img src="{avatar_uri}" width="64" height="64" style="border-radius:50%;" />
                  <div style="
                      background-color: #444;
                      color: white;
                      padding: 12px 16px;
                      margin-left: 10px;
                      border-radius: 14px;
                      max-width: 75%;
                      line-height: 1.4;
                  ">
                    {content}
                  </div>
                </div>
            """)
        else:
            avatar_uri = f"data:image/jpeg;base64,{bot_avatar_b64}"
            html_chunks.append(f"""
                <div style="
                    display: flex;
                    align-items: center;        /* Avatar zentriert vertikal */
                    justify-content: flex-end;
                    margin: 6px 0;
                ">
                  <div style="
                      background-color: #444;
                      color: white;
                      padding: 12px 16px;
                      margin-right: 10px;
                      border-radius: 14px;
                      max-width: 75%;
                      line-height: 1.4;
                  ">
                    {content}
                  </div>
                  <!-- Bot-Icon jetzt auch 64×64 -->
                  <img src="{avatar_uri}" width="64" height="64" style="border-radius:50%;" />
                </div>
            """)
    return "<div style='font-family: sans-serif;'>" + "\n".join(html_chunks) + "</div>"



# 5. Gradio-Interface aufbauen
css = """
/* Fußzeile, API-Leiste und Einstellungs-Icon ausblenden */
footer { display: none !important; }
.gradio_api { display: none !important; }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# 🌸 Mahabharata-Gita RAG-Chatbot")
    gr.Markdown(
        "Ask the chatbot a question about the Bhagavad Gita. "
        "It will search the text, retrieve relevant passages, and then answer your question."
    )

    # (A) Anzeige-Element (HTML) für den gesamten Chatverlauf
    chat_display = gr.HTML(
        "<div style='color: grey; text-align: center; margin-top: 20px;'>"
        "The chat history will appear here…</div>"
    )

    # (B) Textfeld für die neue Frage
    txt = gr.Textbox(
        placeholder="Type your question here…",
        label="Your Question:",
        lines=1
    )

    # (C) State-Objekt zum Speichern der History-Liste
    state = gr.State([])

    # ENTER oder Button → chat_with_bot aufrufen
    # Hinweis: Wir geben drei Outputs zurück (chat_display, state, txt), damit
    # das Textfeld nach Absenden geleert wird.
    txt.submit(chat_with_bot, inputs=[txt, state], outputs=[chat_display, state, txt])
    send_btn = gr.Button("Submit")
    send_btn.click(chat_with_bot, inputs=[txt, state], outputs=[chat_display, state, txt])

    gr.Markdown("---")
    gr.Markdown("🛑 To stop the app, press CTRL+C in the terminal.")

# 6. Gradio-Server starten und Standard-Browser öffnen
if __name__ == "__main__":
    demo.queue()

    # Kurze Verzögerung, damit der Server initialisiert ist (optional)
    # webbrowser.open ruft das System‐Default auf (Chrome, Brave oder jeder andere),
    # je nachdem, was aktuell als Standard‐Browser eingestellt ist.
    webbrowser.open("http://localhost:7860")

    demo.launch(server_name="0.0.0.0", server_port=7860)
