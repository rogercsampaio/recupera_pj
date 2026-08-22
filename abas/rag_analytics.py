import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import streamlit as st

# ============================================================
# CONFIGURAÇÃO
# ============================================================
RAIZ_PROJETO = Path(__file__).resolve().parent.parent

# Aponta para a pasta metricas_rag localizada na raiz
DIR_METRICAS_RAG = RAIZ_PROJETO / "metricas_rag"


def render_rag_analtycs():
  st.title("🔮 Métricas RAG")
  st.caption("Dashboard de Análise e Volumetria da Base de Documentos")
  st.markdown("---")

  caminho_csv = DIR_METRICAS_RAG / "metricas_estatistica_palavras.csv"

  try:
    # Leitura do CSV
    df = pd.read_csv(caminho_csv)

    # ------------------------------------------------------------
    # 1, 2 e 3. INDICADORES PRINCIPAIS (KPIs)
    # ------------------------------------------------------------
    total_docs = len(df)
    total_tokens = df["qtd_tokens"].sum() if "qtd_tokens" in df else 0
    total_palavras = df["qtd_palavras"].sum() if "qtd_palavras" in df else 0

    col1, col2, col3 = st.columns(3)
    with col1:
      st.metric(
          label="📄 Total de Documentos",
          value=f"{total_docs:,}".replace(",", "."),
      )
    with col2:
      st.metric(
          label="🔤 Total de Tokens",
          value=f"{total_tokens:,}".replace(",", "."),
      )
    with col3:
      st.metric(
          label="📝 Total de Palavras",
          value=f"{total_palavras:,}".replace(",", "."),
      )

    st.markdown("---")

    # ------------------------------------------------------------
    # 4. GRÁFICO DE BARRA DUPLA (Palavras x Frases por Documento)
    # ------------------------------------------------------------
    st.subheader("📊 Métricas por Documento")

    fig_barras = go.Figure()
    fig_barras.add_trace(
        go.Bar(
            x=df["nome_arquivo"],
            y=df["qtd_palavras"],
            name="Qtd. Palavras",
            marker_color="#1f77b4",
        )
    )
    fig_barras.add_trace(
        go.Bar(
            x=df["nome_arquivo"],
            y=df["qtd_frases"],
            name="Qtd. Frases",
            marker_color="#ff7f0e",
        )
    )

    fig_barras.update_layout(
        barmode="group",
        xaxis_title="Nome do Arquivo",
        yaxis_title="Quantidade",
        legend_title="Métrica",
        xaxis_tickangle=-45,
        height=450,
    )
    st.plotly_chart(fig_barras, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------------
    # 5. HISTOGRAMA COM DISTRIBUIÇÃO NORMAL & ESTATÍSTICAS
    # ------------------------------------------------------------
    st.subheader("📈 Distribuição de Tokens e Estatísticas")

    tokens = df["qtd_tokens"].dropna()

    # Cálculo da curva Normal
    mu, std = tokens.mean(), tokens.std()
    x_norm = np.linspace(tokens.min(), tokens.max(), 100)
    # Ajusta a escala para sobrepor ao histograma
    bins_count = 15
    y_norm = (
        norm.pdf(x_norm, mu, std)
        * len(tokens)
        * (tokens.max() - tokens.min())
        / bins_count
    )

    # Gráfico Histograma + Curva Normal
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=tokens,
            nbinsx=bins_count,
            name="Distribuição de Tokens",
            marker_color="#2ca02c",
            opacity=0.7,
        )
    )
    fig_hist.add_trace(
        go.Scatter(
            x=x_norm,
            y=y_norm,
            mode="lines",
            name="Distribuição Normal",
            line=dict(color="red", width=2, dash="dash"),
        )
    )

    fig_hist.update_layout(
        xaxis_title="Quantidade de Tokens",
        yaxis_title="Frequência",
        height=400,
    )

    col_graph, col_stats = st.columns([2, 1])

    with col_graph:
      st.plotly_chart(fig_hist, use_container_width=True)

    # Tabela descritiva das estatísticas dos tokens
    with col_stats:
      st.markdown("**Estatísticas de Tokens**")

      q1 = tokens.quantile(0.25)
      mediana = tokens.median()
      q3 = tokens.quantile(0.75)

      df_stats = pd.DataFrame({
          "Métrica": [
              "Mínimo",
              "Máximo",
              "Média",
              "Mediana",
              "Q1 (25%)",
              "Q3 (75%)",
          ],
          "Valor": [
              f"{tokens.min():,.0f}".replace(",", "."),
              f"{tokens.max():,.0f}".replace(",", "."),
              f"{mu:,.2f}".replace(",", "."),
              f"{mediana:,.2f}".replace(",", "."),
              f"{q1:,.2f}".replace(",", "."),
              f"{q3:,.2f}".replace(",", "."),
          ],
      })
      st.dataframe(df_stats, hide_index=True, use_container_width=True)

  except Exception as e:
    st.error(f"Erro ao carregar ou processar as métricas do RAG: {e}")

