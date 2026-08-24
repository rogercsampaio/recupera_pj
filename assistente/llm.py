import os
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
import streamlit as st

# ============================================================
# CONFIGURAÇÃO
# ============================================================

MODELO_GEMINI = "gemini-3.1-flash-lite"

# ============================================================
# CRIAÇÃO DA LLM
# ============================================================


def carregar_llm():
    try:
        # Tenta obter a chave do Streamlit Secrets ou de variável de ambiente
        api_key = None
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
        else:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "A chave 'GEMINI_API_KEY' não foi encontrada em"
                " .streamlit/secrets.toml nem nas variáveis de ambiente."
            )

        # DEFINE AS VARIÁVEIS DE AMBIENTE QUE A SDK DO GOOGLE E O LANGCHAIN EXIGEM
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["GEMINI_API_KEY"] = api_key

        logger.info(f"Inicializando LLM Gemini: {MODELO_GEMINI}")

        llm = ChatGoogleGenerativeAI(
            model=MODELO_GEMINI,
            google_api_key=api_key,
            temperature=0,
        )

        logger.info("LLM Gemini inicializada com sucesso.")
        return llm

    except Exception as excecao:
        logger.exception(f"Erro ao inicializar LLM Gemini: {excecao}")
        raise


# ============================================================
# CARREGAR LLM COM TOOLS
# ============================================================


def carregar_llm_com_tools(llm, tools):
    try:
        logger.info(
            f"Configurando Tool Calling com {len(tools)} ferramenta(s)."
        )

        llm_com_tools = llm.bind_tools(tools)

        logger.info("LLM configurada com Tool Calling com sucesso.")

        return llm_com_tools
    except Exception as excecao:
        logger.exception(f"Erro ao configurar Tool Calling: {excecao}")
        raise


# Inicialização
llm = carregar_llm()