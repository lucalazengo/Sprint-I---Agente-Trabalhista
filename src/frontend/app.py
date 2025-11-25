import sys
import os
from pathlib import Path

# Add project root to python path to allow imports from src
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

import streamlit as st
from src.backend.agent import LaborLawAgent
from src.utils import config

# Page Config
st.set_page_config(
    page_title="Agente Trabalhista",
    page_icon="💬",
    layout="centered"
)

# --- CSS Personalizado ---
st.markdown("""
<style>
    /* Fundo geral (Cinza claro típico de apps de chat) */
    .stApp {
        background-color: #ECE5DD;
    }
    
    /* Cabeçalho (Verde WhatsApp) */
    header[data-testid="stHeader"] {
        background-color: #075E54;
    }
    
    /* Ajuste do título para parecer um Header de App */
    .app-header {
        background-color: #075E54;
        padding: 15px;
        color: white;
        border-radius: 0 0 10px 10px;
        margin-top: -50px; /* Gambiarra para subir sobre o padding padrão */
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        display: flex;
        align_items: center;
        gap: 10px;
    }
    
    /* --- 1. Container da Mensagem (Linha) --- */
    .stChatMessage {
        background-color: transparent !important; /* Remove fundo do container pai */
        box-shadow: none !important;
        border: none !important;
        padding: 0 !important;
        margin-bottom: 15px;
        display: flex;
        align-items: flex-end; /* Alinha avatar na base se houver */
    }

    /* --- 2. Balão de Texto (Conteúdo) --- */
    div[data-testid="stChatMessageContent"] {
        padding: 12px 16px;
        border-radius: 10px;
        max-width: 100%; /* O container pai já limita, mas garantimos */
        box-shadow: 0 1px 2px rgba(0,0,0,0.15);
        width: fit-content;
        position: relative;
    }

    /* --- 3. Estilo do Texto --- */
    div[data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessageContent"] div {
        color: #000000 !important;
        margin: 0;
        font-family: sans-serif;
        line-height: 1.5;
    }

    /* --- 4. Lógica de Cores e Alinhamento --- */
    
    /* ASSISTENTE (Ímpar) -> Balão Branco, Alinhado à Esquerda */
    div[data-testid="stChatMessage"]:nth-of-type(odd) {
        flex-direction: row;
    }
    div[data-testid="stChatMessage"]:nth-of-type(odd) div[data-testid="stChatMessageContent"] {
        background-color: #FFFFFF;
        border-top-left-radius: 0;
        margin-left: 10px; /* Espaço do avatar */
    }

    /* USUÁRIO (Par) -> Balão Verde, Alinhado à Direita */
    div[data-testid="stChatMessage"]:nth-of-type(even) {
        flex-direction: row-reverse;
    }
    div[data-testid="stChatMessage"]:nth-of-type(even) div[data-testid="stChatMessageContent"] {
        background-color: #DCF8C6;
        border-top-right-radius: 0;
        margin-right: 10px; /* Espaço do avatar */
    }
    
    /* Esconder ícones de 'copy' e avatar padrão se desejar limpar mais (opcional, mantendo avatars por enquanto) */

    /* Input de texto (Estilo barra inferior) */
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    
</style>
""", unsafe_allow_html=True)

def initialize_agent():
    """Lazy load the agent only when needed"""
    if "agent" not in st.session_state:
        with st.spinner("Conectando ao Agente Jurídico..."):
            try:
                st.session_state.agent = LaborLawAgent()
            except Exception as e:
                st.error(f"Falha de conexão: {e}")
                st.stop()

def main():
    # Header Customizado
    st.markdown("""
        <div class="app-header">
            <h2>⚖️ Agente Trabalhista</h2>
            <p style='font-size: 0.8em; opacity: 0.8;'>Online • Especialista CLT</p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize Agent
    initialize_agent()

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Olá! Sou seu assistente jurídico. Pode perguntar sobre CLT, férias, horas extras, etc."
        }]

    # Display Chat
    # Iteramos e aplicamos lógica customizada se necessário, mas o CSS já faz o trabalho pesado nos st.chat_message
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input Area
    if prompt := st.chat_input("Mensagem"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Mostra mensagem do usuário imediatamente (o CSS vai alinhar à direita)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Resposta do Assistente
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("⏳ *Digitando...*")

            # Format history
            history = [
                msg for msg in st.session_state.messages 
                if msg["role"] in ("user", "assistant")
            ][:-1]

            # Run Agent
            response, log_trace = st.session_state.agent.run(prompt, history)

            # Display Log (Debug) is DISABLED for Production MVP
            # if log_trace:
            #     with st.expander("🔍 Detalhes da Pesquisa"):
            #         st.code(log_trace, language='text')

            placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
