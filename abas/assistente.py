import streamlit as st
from uteis.extratores import get_usd_brl
from assistente.assistente import executar_agente
from assistente.llm import llm
from assistente.tools import (
    consultar_cliente,
    buscar_historico_interacoes,
    buscar_documentacao,
    consultar_score_regularizacao,
    recomendar_acoes
)

# =====================================================================
# CONFIGURAÇÕES
# =====================================================================

#PRECO_INPUT_TOKEN = 1.50 / 1_000_000
#PRECO_OUTPUT_TOKEN = 9.00 / 1_000_000
#MODEL_NAME = "gemini-3.1-flash-lite"


def render_assistente_page():

    st.title("🤖 Assistente Autônomo de IA")
    st.write(
        "Consulte informações de clientes e interaja "
        "com o assistente utilizando linguagem natural."
    )

    pergunta = st.chat_input(
        "Digite sua pergunta..."
    )

    if pergunta:

        st.chat_message("user").write(
            pergunta
        )

        with st.chat_message("assistant"):

            with st.spinner("Consultando..."):

                try:
                    tools = [
                    consultar_cliente,
                    buscar_historico_interacoes,
                    buscar_documentacao,
                    consultar_score_regularizacao,
                    recomendar_acoes]

                    resposta = executar_agente(
                        pergunta=pergunta,
                        llm=llm,
                        tools=tools
                    )

                    if hasattr(resposta, "content"):
                        conteudo = resposta.content

                        if isinstance(conteudo, list):

                            textos = []

                            for item in conteudo:

                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "text"
                                ):
                                    textos.append(
                                        item.get("text", "")
                                    )

                            conteudo = "\n".join(
                                textos
                            )

                        st.markdown(
                            conteudo
                        )

                    else:

                        st.markdown(
                            str(resposta)
                        )

                except Exception as excecao:

                    st.error(
                        f"Erro ao executar o assistente: "
                        f"{excecao}"
                    )
    # Nota de rodapé sobre os dados
    st.caption(
        "⚠️ *Nota: O RENOVA PJ é uma IA e pode cometer erros.*"
     )
