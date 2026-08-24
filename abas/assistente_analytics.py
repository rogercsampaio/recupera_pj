import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st


def localizar_arquivo(nome_arquivo, pasta_relativa=""):
    """
    Busca robusta por arquivos no Streamlit Cloud / Local.
    Verifica localização do script, diretório de trabalho atual e busca recursiva se necessário.
    """
    # 1. Tenta a partir do diretório do próprio arquivo atual
    dir_script = Path(__file__).resolve().parent
    candidatos = [
        dir_script / pasta_relativa / nome_arquivo,
        dir_script.parent / pasta_relativa / nome_arquivo,
        dir_script.parent.parent / pasta_relativa / nome_arquivo,
        Path(os.getcwd()) / pasta_relativa / nome_arquivo,
    ]

    for cand in candidatos:
        cand_norm = cand.resolve()
        if cand_norm.exists():
            return cand_norm

    # 2. Busca recursiva (Fallback para o Streamlit Cloud /mount/src/...)
    raiz_busca = Path(os.getcwd()).resolve()
    for arq in raiz_busca.rglob(nome_arquivo):
        return arq.resolve()

    return None


def render_assistente_analytics_page():
    st.title("🤖 Analytics PAGE Assistente")

    # ==============================================================================
    # 1. RESOLUÇÃO DOS CAMINHOS
    # ==============================================================================
    caminho_golden = localizar_arquivo("golden_set_completo.csv", "bases_tratadas")
    caminho_metricas = localizar_arquivo("metricas_agente_autonomo.csv", "resultados")

    # ==============================================================================
    # 2. LEITURA DOS ARQUIVOS E TRATAMENTO DE ERRO
    # ==============================================================================
    total_golden = 0
    if caminho_golden and caminho_golden.exists():
        df_golden_count = pd.read_csv(caminho_golden)
        total_golden = len(df_golden_count)

    if not caminho_metricas or not caminho_metricas.exists():
        caminho_esperado = Path(os.getcwd()) / "resultados" / "metricas_agente_autonomo.csv"
        st.error(
            f"⚠️ Arquivo de métricas não encontrado no servidor!\n\n"
            f"**Caminho esperado:** `{caminho_esperado}`\n\n"
            f"**Possíveis causas no Streamlit Cloud:**\n"
            f"1. O arquivo `resultados/metricas_agente_autonomo.csv` não foi commitado no Git (verifique o `.gitignore`).\n"
            f"2. O nome da pasta no GitHub está com maiúscula/minúscula diferente (ex: `Resultados` vs `resultados`)."
        )
        st.info("Execute o notebook de avaliação para gerar/commitar este arquivo no repositório.")
        return

    df_metricas = pd.read_csv(caminho_metricas)

    # ==============================================================================
    # 3. CARDS SUPERIORES: MÉTRICA GERAL & TOTAL DE ITENS
    # ==============================================================================
    st.subheader("📌 Resumo Geral do Desempenho")

    col_total, col_f, col_r, col_cp, col_cr = st.columns(5)

    with col_total:
        st.metric(
            label="Total Avaliado (Golden Set)", value=f"{total_golden} itens"
        )

    with col_f:
        st.metric(
            label="Fidelidade (Faithfulness)",
            value=f"{df_metricas['faithfulness'].mean():.2f}",
        )

    with col_r:
        st.metric(
            label="Relevância da Resposta",
            value=f"{df_metricas['answer_relevancy'].mean():.2f}",
        )

    with col_cp:
        st.metric(
            label="Precisão do Contexto",
            value=f"{df_metricas['context_precision'].mean():.2f}",
        )

    with col_cr:
        st.metric(
            label="Revogação do Contexto (Recall)",
            value=f"{df_metricas['context_recall'].mean():.2f}",
        )

    st.divider()

    # ==============================================================================
    # 4. TABELA EXPLICATIVA DAS MÉTRICAS
    # ==============================================================================
    st.subheader("📚 Detalhamento das Métricas de Avaliação")

    dados_explicacao = [
        {
            "Métrica (Original)": "faithfulness",
            "Métrica (Português)": "Fidelidade / Aderência",
            "Valor Médio": f"{df_metricas['faithfulness'].mean():.4f}",
            "Intervalo Esperado": "0.0 a 1.0",
            "Explicação": (
                "Mede se a resposta gerada é 100% baseada no contexto recuperado,"
                " garantindo que o modelo não alucinou nem inventou fatos."
            ),
        },
        {
            "Métrica (Original)": "answer_relevancy",
            "Métrica (Português)": "Relevância da Resposta",
            "Valor Médio": f"{df_metricas['answer_relevancy'].mean():.4f}",
            "Intervalo Esperado": "0.0 a 1.0",
            "Explicação": (
                "Avalia se a resposta responde diretamente o que foi solicitado"
                " na pergunta, sem desviar do assunto ou dar voltas."
            ),
        },
        {
            "Métrica (Original)": "context_precision",
            "Métrica (Português)": "Precisão do Contexto",
            "Valor Médio": f"{df_metricas['context_precision'].mean():.4f}",
            "Intervalo Esperado": "0.0 a 1.0",
            "Explicação": (
                "Mede a proporção de informação útil presente no contexto"
                " recuperado em relação ao ruído trazido da base."
            ),
        },
        {
            "Métrica (Original)": "context_recall",
            "Métrica (Português)": "Revogação / Cobertura (Recall)",
            "Valor Médio": f"{df_metricas['context_recall'].mean():.4f}",
            "Intervalo Esperado": "0.0 a 1.0",
            "Explicação": (
                "Avalia se o contexto recuperado foi suficiente e cobriu todos os"
                " fatos vitais presentes no Ground Truth."
            ),
        },
    ]

    df_explicacao = pd.DataFrame(dados_explicacao)
    st.dataframe(df_explicacao, use_container_width=True, hide_index=True)

    st.divider()

    # ==============================================================================
    # 5. GRÁFICOS ANALÍTICOS (PLOTLY)
    # ==============================================================================
    col_graf1, col_graf2 = st.columns(2)

    with col_graf1:
        st.subheader("📊 Médias das Métricas por Origem de Dados")

        df_agrupado = (
            df_metricas.groupby("origem")[
                [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                ]
            ]
            .mean()
            .reset_index()
        )

        df_melted = df_agrupado.melt(
            id_vars="origem", var_name="Métrica", value_name="Nota Média"
        )

        fig_metricas = px.bar(
            df_melted,
            x="origem",
            y="Nota Média",
            color="Métrica",
            barmode="group",
            title="Desempenho RAG por Origem de Dados",
            range_y=[0, 1.1],
            text_auto=".2f",
        )
        fig_metricas.update_layout(xaxis_title="Origem", yaxis_title="Nota (0-1)")
        st.plotly_chart(fig_metricas, use_container_width=True)

    with col_graf2:
        st.subheader("📦 Quantidade de Itens Avaliados por Origem")

        df_origem_count = (
            df_metricas["origem"].value_counts().reset_index()
        )
        df_origem_count.columns = ["origem", "Quantidade"]

        fig_origem = px.pie(
            df_origem_count,
            names="origem",
            values="Quantidade",
            title="Distribuição Percentual da Base por Origem",
            hole=0.4,
        )
        fig_origem.update_traces(textinfo="value+percent")
        st.plotly_chart(fig_origem, use_container_width=True)