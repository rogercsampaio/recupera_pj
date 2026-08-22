import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st


def render_assistente_analytics_page():
  st.title("🤖 Analytics PAGE Assistente")

  # ==============================================================================
  # 1. RESOLUÇÃO DINÂMICA DO CAMINHO (Encontra a pasta ds3_mlai_test)
  # ==============================================================================
  caminho_atual = Path(os.getcwd()).resolve()

  # Busca a pasta 'ds3_mlai_test' subindo a árvore de diretórios caso necessário
  if "ds3_mlai_test" in caminho_atual.parts:
    idx = caminho_atual.parts.index("ds3_mlai_test")
    RAIZ_PROJETO = Path(*caminho_atual.parts[: idx + 1])
  else:
    # Fallback padrão tentando subir 1 ou usar o diretório atual
    RAIZ_PROJETO = (
        caminho_atual.parent
        if caminho_atual.name != "ds3_mlai_test"
        else caminho_atual
    )

  CAMINHO_GOLDEN = RAIZ_PROJETO / "bases_tratadas" / "golden_set_completo.csv"
  CAMINHO_METRICAS = RAIZ_PROJETO / "resultados" / "metricas_agente_autonomo.csv"

  # ==============================================================================
  # 2. LEITURA DOS ARQUIVOS
  # ==============================================================================
  # Total de itens no Golden Set
  total_golden = 0
  if CAMINHO_GOLDEN.exists():
    df_golden_count = pd.read_csv(CAMINHO_GOLDEN)
    total_golden = len(df_golden_count)

  # Leitura das Métricas Avaliadas
  if not CAMINHO_METRICAS.exists():
    st.error(
        f"⚠️ Arquivo de métricas não encontrado em: `{CAMINHO_METRICAS}`"
    )
    st.info("Execute o notebook de avaliação para gerar este arquivo.")
    return

  df_metricas = pd.read_csv(CAMINHO_METRICAS)

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