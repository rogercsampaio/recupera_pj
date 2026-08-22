import streamlit as st
from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI

# ============================================================
# CONFIGURAÇÃO
# ============================================================

MODELO_GEMINI = "gemini-3.1-flash-lite"

# ============================================================
# CRIAÇÃO DA LLM
# ============================================================

def carregar_llm():
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        # alterar depois a exposicao
        if not api_key:
            raise ValueError(
                "A chave 'GEMINI_API_KEY' não foi encontrada "
                "em .streamlit/secrets.toml."
            )

        logger.info(f"Inicializando LLM Gemini: {MODELO_GEMINI}")

        llm = ChatGoogleGenerativeAI(
            model=MODELO_GEMINI,
            google_api_key=api_key,
            temperature=0
        )

        logger.info("LLM Gemini inicializada com sucesso.")
        return llm

    except Exception as excecao:
        logger.exception(
            f"Erro ao inicializar LLM Gemini: {excecao}"
        )
        raise



# ============================================================
# CARREGAR LLM COM TOOLS
# ============================================================
def carregar_llm_com_tools(llm, tools):
    try:
        logger.info(
            f"Configurando Tool Calling com "
            f"{len(tools)} ferramenta(s)."
        )

        llm_com_tools = llm.bind_tools(tools)

        logger.info(
            "LLM configurada com Tool Calling com sucesso."
        )

        return llm_com_tools
    except Exception as excecao:
        logger.exception(
            f"Erro ao configurar Tool Calling: {excecao}"
        )
        raise
llm = carregar_llm()