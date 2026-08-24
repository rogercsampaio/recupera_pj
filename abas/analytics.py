import streamlit as st
import pandas as pd
from loguru import logger

# Importa as funções do backend criadas no analytics.py
from analytics.analytics import (
    carregar_e_cruzar_dados,
    aplicar_filtros,
    calcular_indicadores_kpi,
    gerar_grafico_uf_regularizacao,
    gerar_grafico_canal_preferencial,
    obter_tabela_detalhada,
)


def render_analytics_page():
    logger.info("Iniciando renderização da página de Analytics no Streamlit.")
    st.title("📊 Analytics de Regularização PJ")
    st.markdown(
        "Explore o perfil dos clientes e as predições do modelo de propensão à regularização para direcionar as estratégias de cobrança."
    )

    # 1. Carregamento dos dados com cache do Streamlit
    @st.cache_data(show_spinner="Carregando base de clientes e inferências...")
    def load_data():
        return carregar_e_cruzar_dados()

    try:
        df_raw = load_data()
    except Exception as e:
        logger.error(f"Erro ao carregar dados na interface: {e}")
        st.error(f"Erro ao carregar as bases de dados: {e}")
        return

    # 2. Painel de Filtros na Sidebar
    st.sidebar.header("🔍 Filtros Analíticos")

    # Filtro por ID de Cliente
    input_id_cliente = st.sidebar.text_input(
        "ID do Cliente",
        value="",
        placeholder="Digite o ID exato...",
        help="Permite buscar um cliente específico pelo ID.",
    )

    st.sidebar.markdown("---")

    # Filtros Categóricos
    setores_opt = sorted(df_raw["setor"].dropna().unique().tolist())
    portes_opt = sorted(df_raw["porte"].dropna().unique().tolist())
    riscos_opt = sorted(df_raw["risco_setorial"].dropna().unique().tolist())
    ufs_opt = sorted(df_raw["uf"].dropna().unique().tolist())
    canais_opt = sorted(df_raw["canal_preferencial"].dropna().unique().tolist())

    sel_setores = st.sidebar.multiselect("Setor", options=setores_opt)
    sel_portes = st.sidebar.multiselect("Porte", options=portes_opt)
    sel_riscos = st.sidebar.multiselect("Nível de Risco Setorial", options=riscos_opt)
    sel_ufs = st.sidebar.multiselect("Estado (UF)", options=ufs_opt)
    sel_canais = st.sidebar.multiselect("Canal Preferencial", options=canais_opt)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Métricas Quantitativas")

    # Filtros Quantitativos (Sliders dinâmicos baseados no min/max da base)
    range_atraso = st.sidebar.slider(
        "Dias de Atraso",
        min_value=int(df_raw["dias_atraso"].min()),
        max_value=int(df_raw["dias_atraso"].max()),
        value=(int(df_raw["dias_atraso"].min()), int(df_raw["dias_atraso"].max())),
    )

    range_saldo = st.sidebar.slider(
        "Saldo Devedor (R$)",
        min_value=float(df_raw["saldo_devedor"].min()),
        max_value=float(df_raw["saldo_devedor"].max()),
        value=(float(df_raw["saldo_devedor"].min()), float(df_raw["saldo_devedor"].max())),
        format="R$ %.2f",
    )

    range_relacionamento = st.sidebar.slider(
        "Tempo de Relacionamento (Meses)",
        min_value=int(df_raw["tempo_relacionamento_meses"].min()),
        max_value=int(df_raw["tempo_relacionamento_meses"].max()),
        value=(
            int(df_raw["tempo_relacionamento_meses"].min()),
            int(df_raw["tempo_relacionamento_meses"].max()),
        ),
    )

    # 3. Aplicação dos filtros (incluindo o filtro por id_cliente)
    df_filtrado = aplicar_filtros(
        df_raw,
        id_cliente=input_id_cliente,
        setor=sel_setores,
        porte=sel_portes,
        risco_setorial=sel_riscos,
        uf=sel_ufs,
        canal_preferencial=sel_canais,
        range_dias_atraso=range_atraso,
        range_saldo_devedor=range_saldo,
        range_tempo_relacionamento=range_relacionamento,
    )

    # 4. Exibição dos KPIs
    kpis = calcular_indicadores_kpi(df_filtrado)

    col1, col2, col3 = st.columns(3)
    col1.metric(
        label="Total de Clientes (Filtrados)",
        value=f"{kpis['total_clientes']:,}".replace(",", "."),
    )
    col2.metric(
        label="Tendem a Regularizar (Próx. Mês)",
        value=f"{kpis['qtd_tende_regularizar']:,}".replace(",", "."),
        delta=f"{kpis['pct_tende_regularizar']}% do total",
    )
    col3.metric(
        label="Não Tendem a Regularizar",
        value=f"{kpis['qtd_nao_tende_regularizar']:,}".replace(",", "."),
        delta=f"-{kpis['pct_nao_tende_regularizar']}% do total",
        delta_color="inverse",
    )

    st.markdown("---")

    # 5. Renderização dos Gráficos em Colunas
    st.subheader("📈 Visões Analíticas de Propensão")
    g_col1, g_col2 = st.columns(2)

    with g_col1:
        fig_uf = gerar_grafico_uf_regularizacao(df_filtrado)
        st.plotly_chart(fig_uf, use_container_width=True)

    with g_col2:
        fig_canal = gerar_grafico_canal_preferencial(df_filtrado)
        st.plotly_chart(fig_canal, use_container_width=True)

    st.markdown("---")

    # 6. Tabela Detalhada com Opção de Download
    st.subheader("📋 Tabela Detalhada de Clientes e Inferências")

    tab_data = obter_tabela_detalhada(df_filtrado)
    st.dataframe(tab_data, use_container_width=True, hide_index=True)

    # Botão para exportar CSV da visão filtrada
    csv = tab_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Exportar Dados Filtrados (CSV)",
        data=csv,
        file_name="base_analytics_filtrada.csv",
        mime="text/csv",
    )
