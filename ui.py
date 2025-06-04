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
# 0. AVATAR-BILDER PER BASE64 EINLESEN (aus Code B, verbessert)
user_avatar_path = Path("images") / "user_avatar.jpg"
bot_avatar_path  = Path("images") / "bot_avatar.jpg"

def encode_image_to_base64(path: Path) -> str:
    """Liest eine Bilddatei ein und gibt sie als Base64-String mit MIME-Typ zurück, oder Fallback-SVG."""
    if not path.exists():
        print(f"⚠️ Avatar file not found: {path}. Using fallback SVG.")
        if "user" in str(path).lower(): # Unterscheidung für Fallback
            return "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSIjY2ZjZmNmIiBzdHJva2U9IiNjZmNmY2YiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0xOCAyMHYtMS44YzAtMS43LTEuMi0zLjItMy0zLjJoLTYtMS44IDAtMy4yIDEuMy0zLjIgMy4yVjIwIi8+PGNpcmNsZSBjeD0iMTIiIGN5PSI4IiByPSI0Ii8+PC9zdmc+"
        else: # Bot oder generisch
            return "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSIjY2ZjZmNmIiBzdHJva2U9IiNjZmNmY2YiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik00IDdoM2EyIDIgMCAwIDEgMiAydjRjMCAxLjEtLjkgMi0yIDJINGEyIDIgMCAwIDEtMi0yVjFhMiAyIDAgMCAxIDItMmg0Ii8+PHBhdGggZD0ibTEyIDEyIDEgNSIvPjxyZWN0IHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCIgeD0iOSIgeT0iNyIgcnk9IjIiLz48cGF0aCBkPSJtOCA0IDEuNSAxLjVMOCAxMCIvPjxwYXRoIGQ9Im0xNiA0LTEuNSAxLjVMMTYgMTAiLz48L3N2Zz4="
    try:
        data = path.read_bytes()
        base64_encoded_data = base64.b64encode(data).decode("utf-8")
        mime_type = "image/jpeg" if path.suffix.lower() in [".jpg", ".jpeg"] else \
                    "image/png" if path.suffix.lower() == ".png" else \
                    "image/gif" if path.suffix.lower() == ".gif" else \
                    "application/octet-stream"
        return f"data:{mime_type};base64,{base64_encoded_data}"
    except Exception as e:
        print(f"❌ Error encoding image {path}: {e}. Using fallback SVG.")
        # Fallback basierend auf Dateinamensteil (redundant zum obigen Check, aber sicherer)
        if "user" in str(path).lower():
            return "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSIjY2ZjZmNmIiBzdHJva2U9IiNjZmNmY2YiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0xOCAyMHYtMS44YzAtMS43LTEuMi0zLjItMy0zLjJoLTYtMS44IDAtMy4yIDEuMy0zLjIgMy4yVjIwIi8+PGNpcmNsZSBjeD0iMTIiIGN5PSI4IiByPSI0Ii8+PC9zdmc+"
        else:
            return "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSIjY2ZjZmNmIiBzdHJva2U9IiNjZmNmY2YiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik00IDdoM2EyIDIgMCAwIDEgMiAydjRjMCAxLjEtLjkgMi0yIDJINGEyIDIgMCAwIDEtMi0yVjFhMiAyIDAgMCAxIDItMmg0Ii8+PHBhdGggZD0ibTEyIDEyIDEgNSIvPjxyZWN0IHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCIgeD0iOSIgeT0iNyIgcnk9IjIiLz48cGF0aCBkPSJtOCA0IDEuNSAxLjVMOCAxMCIvPjxwYXRoIGQ9Im0xNiA0LTEuNSAxLjVMMTYgMTAiLz48L3N2Zz4="

user_avatar_uri = encode_image_to_base64(user_avatar_path)
bot_avatar_uri  = encode_image_to_base64(bot_avatar_path)

# --------------------------------------------------------------------------------
# 1. FAISS-Index laden oder neu erstellen
print("ℹ️ Loading or creating the FAISS index …")
try:
    vectorstore = load_or_create_vectorstore(DATA_PATH, VECTORSTORE_PATH, EMBEDDING_MODEL_NAME, INDEX_BATCH_SIZE)
except Exception as e:
    print(f"✖️ Error loading/creating vectorstore: {e}"); sys.exit(1)

# 2. RAG-Chain initialisieren
print("ℹ️ Initializing the RAG chain …")
try:
    rag_chain = create_rag_chain(vectorstore, OLLAMA_MODEL_NAME)
except Exception as e:
    print(f"✖️ Error initializing RAG chain: {e}"); sys.exit(1)

# --------------------------------------------------------------------------------
# 3. GLOBALER ABBRUCH-FLAG 
cancel_flag = False

def cancel_request():
    """Wird aufgerufen, wenn der Cancel-Button gedrückt wird."""
    global cancel_flag
    print("ℹ️ Cancel request received.")
    cancel_flag = True

# --------------------------------------------------------------------------------
# 4. Callback-Funktion für den Chat 
def chat_with_bot(user_input, history):
    """
    user_input: String – die neue User-Frage
    history:    Liste von {"role": ..., "content": ...} oder []/None
    Yieldet Updates für die Gradio UI.
    """
    global cancel_flag

    if not user_input or not user_input.strip():
        yield render_chat_html(history or []), history or [], "" 
        return

    if history is None:
        history = []

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": "...", "thinking": True}) 
    yield render_chat_html(history), history, "" 

   
    if cancel_flag:
        history.pop() 
        history.append({"role": "assistant", "content": "❌ Request canceled by user before processing."})
        html_out = render_chat_html(history)
        cancel_flag = False  
        yield html_out, history, ""
        return

    try:
       
        output = rag_chain.invoke({"query": user_input})
        answer = output["result"]

        
        if cancel_flag:
            history.pop() 
            history.append({"role": "assistant", "content": "❌ Request canceled by user during processing."})
            html_out = render_chat_html(history)
            cancel_flag = False  
            yield html_out, history, ""
            return

    except Exception as e:
        print(f"✖️ Error invoking RAG chain: {e}")
        answer = "Sorry, I encountered an error processing your request."
       
        if cancel_flag:
            history.pop() 
            history.append({"role": "assistant", "content": "❌ Request canceled by user (error during processing)."})
            html_out = render_chat_html(history)
            cancel_flag = False
            yield html_out, history, ""
            return

    history.pop() 
    history.append({"role": "assistant", "content": answer})
    html_out = render_chat_html(history)
    yield html_out, history, "" 

# --------------------------------------------------------------------------------
# 5. Hilfsfunktion: HTML für Chat (aus Code B)
def render_chat_html(history):
    if not history:
        return "<div class='empty-chat-message'>Ask me anything about the Gita...</div>"
    html_chunks = []
    for msg in history:
        role = msg["role"]
        content = html.escape(msg.get("content", "")).replace("\n", "<br>")
        is_thinking = msg.get("thinking", False)

        if role == "user":
            html_chunks.append(f"""
                <div class="chat-message user-message">
                  <img src="{user_avatar_uri}" class="avatar" alt="User" />
                  <div class="message-bubble">
                    {content}
                  </div>
                </div>""")
        else: 
            if is_thinking:
                bubble_content = """
                    <div class="typing-indicator">
                        <span></span><span></span><span></span>
                    </div>"""
                html_chunks.append(f"""
                    <div class="chat-message bot-message">
                      <div class="message-bubble thinking-bubble">
                        {bubble_content}
                      </div>
                      <img src="{bot_avatar_uri}" class="avatar" alt="Bot" />
                    </div>""")
            else:
                html_chunks.append(f"""
                    <div class="chat-message bot-message">
                      <div class="message-bubble">
                        {content}
                      </div>
                      <img src="{bot_avatar_uri}" class="avatar" alt="Bot" />
                    </div>""")
    return "<div class='chat-messages-container' id='chat-messages-container-id'>" + "\n".join(html_chunks) + "</div>"

# --------------------------------------------------------------------------------
# 6. Gradio-Interface 

chat_ui_css = """
/* CSS aus Code B, erweitert für Cancel-Button */
html, body {
    margin: 0;
    padding: 0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #1e1e1e;
    color: #e0e0e0;
    min-height: 100vh;
}

.gradio-container {
    display: flex !important;
    flex-direction: column !important;
    min-height: 100vh;
    width: 100%;
    background-color: #1e1e1e !important;
}

.gradio-container > footer,
.gradio-container > .header,
.gradio_api {
    display: none !important;
}

#chat-interface-wrapper {
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
    max-width: 900px;
    margin: 0 auto;
    border-left: 1px solid #2a2a2a;
    border-right: 1px solid #2a2a2a;
    background-color: #1e1e1e;
    flex-grow: 1;
}

#header-area {
    padding: 12px 25px;
    border-bottom: 1px solid #333;
    background-color: #252525;
    flex-shrink: 0;
}
#header-area h1 { margin: 0 0 4px 0; font-size: 1.2em; color: #fafafa; font-weight: 600; }
#header-area p { margin: 0; font-size: 0.75em; color: #b0b0b0; }

#chat-display-outer-container {
    flex-grow: 1;
    min-height: 50px; /* Mindesthöhe für den Chat-Anzeigebereich */
    padding: 20px 25px;
    background-color: #1e1e1e;
    overflow-y: auto; /* Wichtig, damit der Chat scrollbar wird */
}

#chat-display-scroll-area { /* Dieser ID ist für den HTML-Inhalt, nicht den Container */
    width: 100%;
}

.empty-chat-message { text-align: center; color: #777; margin-top: 40px; font-style: italic; font-size: 0.95em; }
.chat-messages-container { width: 100%; padding-bottom: 10px; }

.chat-message { display: flex; align-items: flex-end; margin-bottom: 20px; max-width: 100%; }
.avatar {
    width: 36px !important; height: 36px !important;
    min-width: 36px !important; min-height: 36px !important;
    border-radius: 50%; object-fit: cover;
    border: 1px solid #444; flex-shrink: 0;
}
.message-bubble {
    padding: 12px 18px;
    border-radius: 18px;
    line-height: 1.6;
    max-width: 75%;
    word-wrap: break-word;
    box-shadow: 0 3px 8px rgba(0,0,0,0.25);
}
.user-message { justify-content: flex-start; }
.user-message .avatar { margin-right: 12px; }
.user-message .message-bubble { background-color: #303035; color: #e8e8e8; border-bottom-left-radius: 6px; }
.bot-message { justify-content: flex-end; }
.bot-message .avatar { margin-left: 12px; order: 1; }
.bot-message .message-bubble { background-color: #534FBF; color: #ffffff; border-bottom-right-radius: 6px; }
.thinking-bubble { background-color: #2a2a2e !important; padding: 14px 18px !important; }
.typing-indicator { display: flex; align-items: center; justify-content: center; }
.typing-indicator span {
    height: 8px; width: 8px; background-color: #666;
    border-radius: 50%; display: inline-block; margin: 0 3px;
    animation:-gr-typing-indicator-bounce 1.2s infinite ease-in-out both;
}
.typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
.typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
@keyframes -gr-typing-indicator-bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1.0); } }

#input-area-wrapper {
    padding: 12px 25px;
    border-top: 1px solid #333;
    background-color: #252525;
    flex-shrink: 0;
}
#input-row, #input-area-wrapper .gr-row { /* Gilt für die Zeile mit Textbox und Buttons */
    align-items: flex-end !important;
    gap: 10px !important;
}
/* Styling für Textbox-Container (damit er wächst) */
#input-row > div:first-child, /* Gradio umschließt Textbox oft mit einem div */
#input-row > span:first-child > div, /* Manchmal auch so */
#input-row .gr-form { /* Generischer Fallback für den Textbox-Wrapper */
    border: none !important; padding: 0 !important; margin: 0 !important;
    box-shadow: none !important; background-color: transparent !important;
    display: flex !important; /* Wichtig für Flex-Grow */
    flex-grow: 1 !important; /* Lässt Textbox den verfügbaren Platz einnehmen */
}
#input-area-wrapper textarea {
    background-color: #2c2c2e !important; color: #e0e0e0 !important;
    border: 1px solid #444 !important; border-radius: 22px !important;
    padding: 10px 18px !important; /* Angepasst für Höhe */
    box-shadow: none !important;
    min-height: 24px !important; /* Mindesthöhe für eine Zeile */
    line-height: 1.5;
    flex-grow: 1;
    overflow-y: hidden; /* Verhindert Scrollbar, wenn max_lines nicht erreicht */
}
#input-area-wrapper textarea::placeholder { color: #777 !important; }

/* Basis-Button-Styling (wird von Send und Cancel geerbt/überschrieben) */
#input-area-wrapper button.gr-button {
    min-width: auto !important; /* Erlaubt kleinere Buttons */
    padding: 0px 15px !important;
    height: 46px !important; /* Gleiche Höhe wie Textarea mit Padding */
    border-radius: 22px !important;
    border: none !important;
    font-weight: 500;
    transition: background-color 0.2s ease;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    flex-shrink: 0; /* Verhindert, dass Buttons schrumpfen */
    align-self: flex-end; /* Am unteren Rand der Row ausrichten */
}

/* Send-Button (spezifisch aus Code B) */
#send-btn { /* elem_id für den Send-Button */
    background-color: #6A65D7 !important;
    color: white !important;
}
#send-btn:hover {
    background-color: #534FBF !important;
}

/* Cancel-Button (Stile aus Code A, angepasst) */
#cancel-btn { /* elem_id für den Cancel-Button */
    background-color: #dc3545 !important;
    color: white !important;
    font-weight: bold !important; /* Beibehalten, wenn gewünscht */
}
#cancel-btn:hover {
    background-color: #c82333 !important;
}

/* Aktiv/Inaktiv Styling für Send-Button (optional, wenn txt.change verwendet wird) */
#send-btn:disabled {
    background-color: #4a4a50 !important;
    cursor: not-allowed !important;
}


#footer-ctrl-c-info {
    padding: 8px 25px;
    background-color: #252525;
    flex-shrink: 0;
    text-align:center;
    color:#666;
    font-size:0.75em;
    border-top: 1px solid #333;
}
#footer-ctrl-c-info div { padding: 0; margin: 0; }

/* Verstecke Gradio-Fortschrittsanzeigen, da wir eigene "Thinking"-Bubble haben */
.gradio-container .generating,
.gradio-container .status-tracker,
.gradio-container .progress-bar,
.gradio-container .progress-text,
.gradio-container .eta {
    display: none !important;
}
"""

with gr.Blocks(css=chat_ui_css, title="Mahabharata-Gita RAG-Chatbot") as demo:
    with gr.Column(elem_id="chat-interface-wrapper"):
        with gr.Column(elem_id="header-area"):
            gr.Markdown("<h1>🌸 Mahabharata-Gita RAG-Chatbot</h1>")
            gr.Markdown("<p>Ask me anything about the Gita. I'll search the text, retrieve relevant passages, and then answer your question.</p>")

  
        with gr.Column(elem_id="chat-display-outer-container"):
            chat_display_html_component = gr.HTML(
                render_chat_html([]), 
                elem_id="chat-display-scroll-area" 
            )

        with gr.Column(elem_id="input-area-wrapper"):
            with gr.Row(elem_id="input-row", equal_height=False): 
                txt_input = gr.Textbox(
                    show_label=False,
                    placeholder="Type your question...",
                    lines=1,
                    max_lines=5, 
                    container=False, 
                    autofocus=True, 
                    scale=18 
                )
                send_btn = gr.Button(
                    "Send",
                   
                    elem_id="send-btn", 
                    scale=0.2, 
                    interactive=False 
                )
                cancel_btn = gr.Button( 
                    "Cancel",
                    elem_id="cancel-btn",
                    scale=0.1 
                )

        with gr.Column(elem_id="footer-ctrl-c-info"):
             gr.Markdown("<div>🛑 To stop the app, press CTRL+C in the terminal.</div>")


    chat_state = gr.State([])


    txt_input.change(
        fn=lambda value: gr.update(interactive=bool(value and value.strip())),
        inputs=[txt_input],
        outputs=[send_btn]
    )


    txt_input.submit(
        fn=chat_with_bot,
        inputs=[txt_input, chat_state],
        outputs=[chat_display_html_component, chat_state, txt_input],

    )

    send_btn.click(
        fn=chat_with_bot,
        inputs=[txt_input, chat_state],
        outputs=[chat_display_html_component, chat_state, txt_input],

    )


    cancel_btn.click(
        fn=cancel_request,
        inputs=None,
        outputs=None
    )
    cancel_btn.click( 
        fn=lambda: "",
        inputs=None,
        outputs=[txt_input]
    )


# 7. Gradio-Server starten
if __name__ == "__main__":
    print("ℹ️ Starting Gradio app...")
    demo.queue()
    
    local_url = f"http://127.0.0.1:7860" 
    
    print(f"✅ Gradio app will be running at {local_url} or http://0.0.0.0:7860")
    
    try:
        webbrowser.open(local_url)
        print("   Attempting to open in browser...")
    except Exception as e:
        print(f"⚠️ Could not open browser automatically: {e}. Please open {local_url} manually.")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        favicon_path="images/Mahabharata_Favicon.png"
    )