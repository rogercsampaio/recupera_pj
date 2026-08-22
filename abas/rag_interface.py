import streamlit as st
from loguru import logger
from pathlib import Path
from rag.carregamento import carregar_colecao_rag
from rag.busca import buscar_documentos


# ============================================================
# CONFIGURAÇÃO
# ============================================================

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
CAMINHO_CHROMA = RAIZ_PROJETO / "chroma_db"
NOME_COLECAO = "documentos"

# ============================================================
# CARREGAMENTO DA COLEÇÃO
# ============================================================

@st.cache_resource
def carregar_recursos_rag_interface():

    try:

        logger.info(
            "Carregando coleção RAG para a interface."
        )

        client, collection = carregar_colecao_rag(
            caminho_db=CAMINHO_CHROMA,
            nome_colecao=NOME_COLECAO
        )

        return client, collection

    except Exception as excecao:

        logger.exception(
            f"Erro ao carregar recursos da interface RAG: "
            f"{excecao}"
        )

        raise


# ============================================================
# INTERFACE RAG
# ============================================================

def render_rag_page(
    modelo_embedding
):
    """
    Renderiza a interface de busca semântica da RAG.

    A página realiza a recuperação dos documentos utilizando
    embeddings + ChromaDB, sem utilização de LLM.
    """

    st.title(
        "🔎 RAG / Busca Documental"
    )

    st.caption(
        "Busca semântica sobre os documentos utilizados "
        "como evidência pelo Assistente Autônomo de IA."
    )

    st.divider()

    # --------------------------------------------------------
    # Carregar coleção
    # --------------------------------------------------------

    try:

        client, collection = (
            carregar_recursos_rag_interface()
        )

    except Exception:

        st.error(
            "Não foi possível carregar a base documental."
        )

        st.stop()

    # --------------------------------------------------------
    # Informações da coleção
    # --------------------------------------------------------

    quantidade_documentos = collection.count()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "📚 Documentos indexados",
            quantidade_documentos
        )

    with col2:

        st.metric(
            "🗂️ Coleção",
            NOME_COLECAO
        )

    st.divider()

    # --------------------------------------------------------
    # Consulta
    # --------------------------------------------------------

    st.subheader(
        "🔍 Consulta documental"
    )

    consulta = st.text_input(
        "Digite sua pergunta",
        placeholder=(
            "Ex.: Quais são as regras para "
            "parcelamento e desconto?"
        )
    )

    # --------------------------------------------------------
    # Quantidade de documentos
    # --------------------------------------------------------

    n_resultados = st.slider(
        "Quantidade de documentos recuperados",
        min_value=1,
        max_value=min(10, quantidade_documentos),
        value=min(3, quantidade_documentos)
    )

    buscar = st.button(
        "🔎 Buscar documentos",
        type="primary"
    )

    # --------------------------------------------------------
    # Executar busca
    # --------------------------------------------------------

    if buscar:

        if not consulta.strip():

            st.warning(
                "Digite uma consulta para realizar a busca."
            )

            return

        try:

            with st.spinner(
                "Realizando busca semântica..."
            ):

                resultados = buscar_documentos(
                    consulta=consulta,
                    modelo_embedding=modelo_embedding,
                    colecao=collection,
                    n_resultados=n_resultados
                )

            if not resultados:

                st.info(
                    "Nenhum documento foi recuperado."
                )

                return

            # ------------------------------------------------
            # Resultado
            # ------------------------------------------------

            st.success(
                f"{len(resultados)} documento(s) recuperado(s)."
            )

            st.subheader(
                "📚 Evidências recuperadas"
            )

            # ------------------------------------------------
            # Mostrar documentos
            # ------------------------------------------------

            for posicao, documento in enumerate(
                resultados,
                start=1
            ):

                id_documento = documento[
                    "id_documento"
                ]

                similaridade = documento[
                    "similaridade_cosseno"
                ]

                score = documento[
                    "score_similaridade"
                ]

                texto = documento[
                    "texto"
                ]

                with st.expander(
                    f"📄 {posicao}. {id_documento} "
                    f"— Similaridade: {score:.2f}%"
                ):

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Similaridade do cosseno",
                            f"{similaridade:.4f}"
                        )

                    with col2:

                        st.metric(
                            "Score de similaridade",
                            f"{score:.2f}%"
                        )

                    st.markdown(
                        "**Conteúdo recuperado:**"
                    )

                    st.text(
                        texto
                    )

        except Exception as excecao:

            logger.exception(
                f"Erro ao realizar busca na interface RAG: "
                f"{excecao}"
            )

            st.error(
                "Ocorreu um erro durante a busca documental."
            )