import streamlit as st
from pathlib import Path

from abas.analytics import render_analytics_page
from abas.assistente import render_assistente_page
from abas.modelo_preditivo import render_modelo_preditivo_page
from abas.rag_interface import render_rag_page
from abas.rag_analytics import render_rag_analtycs
from abas.assistente_analytics import render_assistente_analytics_page
from abas.analytics import render_analytics_page
from conf.logger_config import setup_logs_once
from conf.proxy_config import configurar_ipv4

from rag.carregamento import (
    carregar_modelo_embedding,
    carregar_colecao_rag
)

def carregar_css():
    css_path = Path("estilo.css")

    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )

carregar_css()
# ============================================================
# RECURSOS DA APLICAÇÃO
# ============================================================

@st.cache_resource
def carregar_recursos_rag():

    modelo_embedding = carregar_modelo_embedding()

    _, colecao = carregar_colecao_rag(
        caminho_db="chroma_db",
        nome_colecao="documentos"
    )

    return modelo_embedding, colecao


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Assistente de Recuperação PJ",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# NAVEGAÇÃO
# ============================================================

def main():

    st.title(
        "🤖 Assistente de Recuperação PJ"
    )

    pagina = st.radio(
        "Navegação",
        [
            "📊 Analytics",
            "🔎 RAG / Documentos",
            "🤖 Assistente Autônomo de IA",
            "🔮 Modelo Preditivo",
            "📊 Métricas RAG",
            "📊 Métricas Assistente"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.divider()

    # --------------------------------------------------------
    # Analytics
    # --------------------------------------------------------

    if pagina == "📊 Analytics":
        render_assistente_analytics_page()

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    elif pagina == "🔎 RAG / Documentos":

        modelo_embedding, colecao = (
            carregar_recursos_rag()
        )

        render_rag_page(
            modelo_embedding=modelo_embedding
        )

    # --------------------------------------------------------
    # Assistente
    # --------------------------------------------------------

    elif pagina == "🤖 Assistente Autônomo de IA":

        render_assistente_page()

    # --------------------------------------------------------
    # Modelo Preditivo
    # --------------------------------------------------------

    elif pagina == "🔮 Modelo Preditivo":

        render_modelo_preditivo_page()
    
    # --------------------------------------------------------
    # Modelo RAG Analytics
    # --------------------------------------------------------
    elif pagina == "📊 Métricas RAG":
    
        render_rag_analtycs()
    
    # --------------------------------------------------------
    # Modelo RAG Analytics
    # --------------------------------------------------------
    elif pagina == "📊 Métricas Assistente":
        render_assistente_analytics_page()


# ============================================================
# INICIALIZAÇÃO
# ============================================================

if __name__ == "__main__":
    setup_logs_once()
    configurar_ipv4()
    main()
