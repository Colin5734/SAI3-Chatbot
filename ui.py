import gradio as gr
import sys
import html
import base64
from pathlib import Path
import webbrowser

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
# 0. AVATAR-BILDER PER BASE64 EINLESEN

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
    print("❌ Fehlende Avatarbilder: Stelle sicher, dass 'images/user_avatar.jpg' und 'images/bot_avatar.jpg' existieren.")
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

# --------------------------------------------------------------------------------
# 3. GLOBALER ABBRUCH-FLAG
cancel_flag = False

def cancel_request():
    """Wird aufgerufen, wenn der Cancel-Button gedrückt wird."""
    global cancel_flag
    cancel_flag = True

# --------------------------------------------------------------------------------
# 4. Callback-Funktion für den Chat (mit History-State)
def chat_with_bot(user_input, history):
    """
    user_input: String – die neue User-Frage
    history:    Liste von {"role": ..., "content": ...} oder []/None beim ersten Aufruf
    """
    global cancel_flag

    # ─── Abbruch bei leerem Input ───────────────────────────────────────────────────
    # Wenn user_input nur aus Leerzeichen (oder komplett leer) besteht, 
    # gib History unverändert zurück und verhindere jede Verarbeitung.
    if not user_input or not user_input.strip():
        # history bleibt gleich, html_out bleibt gleich
        return render_chat_html(history), history, ""
    # ────────────────────────────────────────────────────────────────────────────────

    if history is None:
        history = []

    # LLM-Aufruf (blockierend)
    output = rag_chain.invoke({"query": user_input})
    answer = output["result"]

    # Wenn Cancel während der LLM-Berechnung gedrückt wurde, unterdrücke die echte Antwort:
    if cancel_flag:
        # Alte History beibehalten, aber eine "Request canceled!"-Nachricht hinzufügen:
        history.append({"role": "assistant", "content": "❌ Request canceled!"})
        html_out = render_chat_html(history)
        cancel_flag = False  # Flag zurücksetzen
        return html_out, history, ""  # Textfeld zurücksetzen

    # Wenn nicht abgebrochen, erweitern wir die History normal:
    history.append({"role": "user",      "content": user_input})
    history.append({"role": "assistant", "content": answer})
    html_out = render_chat_html(history)

    return html_out, history, ""  # Textfeld zurücksetzen



# --------------------------------------------------------------------------------
# 5. Hilfsfunktion: Aus kompletter History einen HTML-String bauen

def render_chat_html(history):
    """
    history: Liste von {"role":"user"/"assistant", "content":...}

    Gibt einen einzigen HTML-String zurück, in dem jede Nachricht in einer
    dunkelgrauen Sprechblase mit weißem Text dargestellt wird, plus Avatar.
    """
    html_chunks = []
    for msg in history:
        role    = msg["role"]
        content = html.escape(msg["content"])

        if role == "user":
            # User-Nachricht (links): Avatar links, dunkelgraue Blase
            avatar_uri = f"data:image/jpeg;base64,{user_avatar_b64}"
            html_chunks.append(f"""
                <div style="
                    display: flex;
                    align-items: center;
                    margin: 6px 0;
                ">
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
            # Bot-Nachricht (rechts): Avatar rechts, dunkelgraue Blase
            avatar_uri = f"data:image/jpeg;base64,{bot_avatar_b64}"
            html_chunks.append(f"""
                <div style="
                    display: flex;
                    align-items: center;
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
                  <img src="{avatar_uri}" width="64" height="64" style="border-radius:50%;" />
                </div>
            """)
    return "<div style='font-family: sans-serif;'>" + "\n".join(html_chunks) + "</div>"

# --------------------------------------------------------------------------------
# 6. Gradio-Interface aufbauen (mit History-State)

css = """
/* Verstecke Footer/API-Links */
footer { display: none !important; }
.gradio_api { display: none !important; }

/* Submit-Button: grüner Hintergrund, fette weiße Schrift */
#submit-btn {
    background-color: #28a745 !important;
    color: white !important;
    font-weight: bold !important;
}

/* Hover-Effekt für Submit: etwas dunkleres Grün */
#submit-btn:hover {
    background-color: #1e7e34 !important;
}

/* Cancel-Button: roter Hintergrund, fette weiße Schrift */
#cancel-btn {
    background-color: #dc3545 !important;
    color: white !important;
    font-weight: bold !important;
}

/* Hover-Effekt für Cancel: etwas dunkleres Rot */
#cancel-btn:hover {
    background-color: #c82333 !important;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# 📖 Mahabharata-Gita RAG-Chatbot")
    gr.Markdown(
        "Ask the chatbot a question about the Bhagavad Gita. "
        "It will search the text, retrieve relevant passages, and then answer your question."
    )

    # (A) Anzeige-Element (HTML) für den Chatverlauf
    chat_display = gr.HTML(
        "<div style='color: #888; text-align: center; margin-top: 20px; font-weight: bold !important;'>"
        "The chat history will appear here…</div>"
    )

    # (B) Textbox für die neue Frage (mit Label "Your Question:")
    txt = gr.Textbox(
        placeholder="Type your question here…",
        label="Your Question:",
        lines=1,
        elem_id="txt-input",
        autofocus=True
    )

    # (C) History‐State
    state = gr.State([])

    # (D) Submit‐Button
    send_btn = gr.Button("Submit", elem_id="submit-btn", interactive=False)

    txt.change(
    fn=lambda value: gr.update(interactive=bool(len(value.strip()))),
    inputs=[txt],
    outputs=[send_btn]
)

    # (E) Cancel‐Button
    cancel_btn = gr.Button("Cancel", elem_id="cancel-btn")

    # ENTER → chat_with_bot (mit „processing“-Anzeige nur im Textfeld)
    txt.submit(
        fn=chat_with_bot,
        inputs=[txt, state],
        outputs=[chat_display, state, txt],
        show_progress=[False, False, True]
    )

    # Submit‐Button → chat_with_bot (mit „processing“-Anzeige nur im Textfeld)
    send_btn.click(
        fn=chat_with_bot,
        inputs=[txt, state],
        outputs=[chat_display, state, txt],
        show_progress=[False, False, True]
    )

    # Cancel‐Button setzt Abbruch-Flag und leert das Textfeld
    cancel_btn.click(cancel_request)
    cancel_btn.click(lambda: "", [], [txt])

    gr.Markdown("---")
    gr.Markdown("⛔ To stop the app, press CTRL+C in the terminal.")

# 7. Gradio-Server starten und Standard-Browser öffnen
if __name__ == "__main__":
    demo.queue()
    webbrowser.open("http://localhost:7860")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        favicon_path="images/Mahabharata_Favicon.png"
    )
