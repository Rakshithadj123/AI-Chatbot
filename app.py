import streamlit as st #
import ollama #ollama is a Python library that allows you to interact with the Ollama API, which provides access to various AI models for natural language processing tasks.

# 1. Page Configuration
st.set_page_config(
    page_title="My AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 My AI Chatbot")
st.write("Ask me anything!")

# 2. Sidebar Configuration (Model Selection & Clear Chat)
with st.sidebar:
    st.header("Chat Settings")
    model_name = st.text_input(
        "Ollama Model",
        value="phi3:mini",
        #value = "llama3.2:latest ",
        help="Enter the name of any model downloaded in Ollama (e.g., phi3:mini, llama3.2, mistral)"
    )

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 3. Initialize Conversation Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Render Conversation History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. User Input
prompt = st.chat_input("Type your message...")

if prompt:
    # Display the user's message immediately in the UI
    with st.chat_message("user"):
        st.markdown(prompt)

    # Append user message to conversation history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # 6. Stream Assistant Response with Safe Error Handling
    with st.chat_message("assistant"):
        try:
            # Generator function: yields text chunks from Ollama in real-time
            def generate_response():
                stream = ollama.chat(
                    model=model_name,
                    messages=st.session_state.messages,
                    stream=True
                )
                for chunk in stream:
                    yield chunk["message"]["content"]

            # Streamlit displays words progressively on screen as they arrive
            full_response = st.write_stream(generate_response)

            # Record the full assistant reply in conversation history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

        except Exception as e:
            # Safe rollback: Only remove the last user prompt that failed to get an answer,
            # leaving all earlier conversation turns completely untouched.
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()

            st.error("⚠️ Could not get a response from Ollama.")
            st.info(
                f"**Troubleshooting Checklist:**\n\n"
                f"1. Make sure the Ollama application is running on your PC.\n"
                f"2. Check that the model is downloaded. Open PowerShell and run:\n"
                f"   `ollama pull {model_name}`\n"
                f"3. Verify that the model name in the sidebar matches your installed model."
            )
            st.caption(f"Technical error: {e}")