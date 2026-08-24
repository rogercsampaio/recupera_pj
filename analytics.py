import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger

# Caminhos padrão a partir da raiz do projeto
PATH_CLIENTES = os.path.join("bases_tratadas", "clientes_sem_nulos.csv")
PATH_INFERENCIAS = os.path.join("resultados", "xgboost_inferencia_todos.csv")


def carregar_e_cruzar_dados(
    path_clientes: str = PATH_CLIENTES, path_inferencias: str = PATH_INFERENCIAS
) -> pd.DataFrame:
    """Carrega as bases de clientes e inferências e realiza a junção lado a lado (merge)."""
    logger.info("Iniciando o carregamento das bases para o motor analítico...")

    if not os.path.exists(path_clientes):
        logger.error(f"Arquivo de clientes não encontrado no caminho: {path_clientes}")
        raise FileNotFoundError(
            f"Arquivo de clientes não encontrado em: {path_clientes}"
        )

    if not os.path.exists(path_inferencias):
        if os.path.exists(path_inferencias + ".csv"):
            path_inferencias += ".csv"
        else:
            logger.error(f"Arquivo de inferências não encontrado em: {path_inferencias}")
            raise FileNotFoundError(
                f"Arquivo de inferências não encontrado em: {path_inferencias}"
            )

    logger.debug(f"Carregando clientes de: {path_clientes}")
    df_clientes = pd.read_csv(path_clientes)
    
    logger.debug(f"Carregando inferências de: {path_inferencias}")
    df_inferencias = pd.read_csv(path_inferencias)

    # Caso as duas bases estejam perfeitamente alinhadas por linha
    if len(df_clientes) == len(df_inferencias) and "id_cliente" not in df_clientes.columns:
        logger.info("Realizando junção horizontal direta (concat por linha).")
        df_full = pd.concat([df_clientes, df_inferencias], axis=1)
        df_full = df_full.loc[:, ~df_full.columns.duplicated()]
    else:
        chave = "id_cliente" if "id_cliente" in df_clientes.columns else df_clientes.columns[0]
        logger.info(f"Realizando merge das bases utilizando a chave: '{chave}'.")
        df_full = pd.merge(df_clientes, df_inferencias, on=chave, how="inner")

    logger.success(f"Bases carregadas e cruzadas com sucesso! Total de registros: {len(df_full)}")
    return df_full


def aplicar_filtros(
    df: pd.DataFrame,
    id_cliente: list | str | int = None,  # Novo parâmetro adicionado
    setor: list = None,
    porte: list = None,
    risco_setorial: list = None,
    uf: list = None,
    canal_preferencial: list = None,
    range_tempo_relacionamento: tuple = None,
    range_faturamento: tuple = None,
    range_saldo_devedor: tuple = None,
    range_dias_atraso: tuple = None,
    range_contratos_ativos: tuple = None,
    range_parcelas_vencidas: tuple = None,
) -> pd.DataFrame:
    """Aplica filtros categóricos, identificadores e quantitativos sobre o DataFrame cruzado."""
    logger.info("Aplicando filtros sobre o conjunto de dados...")
    df_filtrado = df.copy()
    total_inicial = len(df_filtrado)

    # Filtro por ID de Cliente
    if id_cliente is not None and id_cliente != "" and id_cliente != []:
        if "id_cliente" in df_filtrado.columns:
            if isinstance(id_cliente, (list, tuple, set)):
                # Normaliza a lista para string/int conforme o tipo da coluna
                ids_busca = [str(i).strip() for i in id_cliente if str(i).strip() != ""]
                df_filtrado = df_filtrado[
                    df_filtrado["id_cliente"].astype(str).isin(ids_busca)
                ]
            else:
                id_str = str(id_cliente).strip()
                df_filtrado = df_filtrado[
                    df_filtrado["id_cliente"].astype(str) == id_str
                ]
        else:
            logger.warning("Coluna 'id_cliente' não encontrada no DataFrame para filtragem.")

    # Filtros Categóricos
    if setor:
        df_filtrado = df_filtrado[df_filtrado["setor"].isin(setor)]
    if porte:
        df_filtrado = df_filtrado[df_filtrado["porte"].isin(porte)]
    if risco_setorial:
        df_filtrado = df_filtrado[df_filtrado["risco_setorial"].isin(risco_setorial)]
    if uf:
        df_filtrado = df_filtrado[df_filtrado["uf"].isin(uf)]
    if canal_preferencial:
        df_filtrado = df_filtrado[
            df_filtrado["canal_preferencial"].isin(canal_preferencial)
        ]

    # Filtros Quantitativos
    if range_tempo_relacionamento:
        df_filtrado = df_filtrado[
            df_filtrado["tempo_relacionamento_meses"].between(*range_tempo_relacionamento)
        ]
    if range_faturamento:
        df_filtrado = df_filtrado[
            df_filtrado["faturamento_mensal_estimado"].between(*range_faturamento)
        ]
    if range_saldo_devedor:
        df_filtrado = df_filtrado[
            df_filtrado["saldo_devedor"].between(*range_saldo_devedor)
        ]
    if range_dias_atraso:
        df_filtrado = df_filtrado[
            df_filtrado["dias_atraso"].between(*range_dias_atraso)
        ]
    if range_contratos_ativos:
        df_filtrado = df_filtrado[
            df_filtrado["qtd_contratos_ativos"].between(*range_contratos_ativos)
        ]
    if range_parcelas_vencidas:
        df_filtrado = df_filtrado[
            df_filtrado["qtd_parcelas_vencidas"].between(*range_parcelas_vencidas)
        ]

    total_final = len(df_filtrado)
    logger.info(f"Filtros aplicados. Registros reduzidos de {total_inicial} para {total_final}.")
    return df_filtrado


def calcular_indicadores_kpi(df: pd.DataFrame) -> dict:
    """Calcula os KPIs de total de clientes e volumetria/percentual de tendência à regularização."""
    logger.info("Calculando indicadores de KPI...")
    total_clientes = len(df)

    if total_clientes == 0:
        logger.warning("DataFrame vazio fornecido para cálculo de KPIs.")
        return {
            "total_clientes": 0,
            "qtd_tende_regularizar": 0,
            "pct_tende_regularizar": 0.0,
            "qtd_nao_tende_regularizar": 0,
            "pct_nao_tende_regularizar": 0.0,
        }

    qtd_tende = int((df["previsao"] == 1).sum())
    qtd_nao_tende = int((df["previsao"] == 0).sum())

    pct_tende = (qtd_tende / total_clientes) * 100
    pct_nao_tende = (qtd_nao_tende / total_clientes) * 100

    logger.debug(f"KPIs calculados - Total: {total_clientes} | Tende: {qtd_tende} ({pct_tende:.1f}%) | Não Tende: {qtd_nao_tende} ({pct_nao_tende:.1f}%)")

    return {
        "total_clientes": total_clientes,
        "qtd_tende_regularizar": qtd_tende,
        "pct_tende_regularizar": round(pct_tende, 2),
        "qtd_nao_tende_regularizar": qtd_nao_tende,
        "pct_nao_tende_regularizar": round(pct_nao_tende, 2),
    }


def obter_tabela_detalhada(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna a tabela completa filtrada pronta para exibição no Streamlit."""
    logger.debug(f"Retornando tabela detalhada com {len(df)} registros.")
    return df


# ==============================================================================
# FUNÇÕES DE GERAÇÃO DE GRÁFICOS (PLOTLY)
# ==============================================================================

def gerar_grafico_uf_regularizacao(df: pd.DataFrame) -> go.Figure:
    """Gráfico 4: Volume e Proporção de Clientes Propensos a Regularizar por UF."""
    logger.info("Gerando gráfico de regularização por UF...")
    if df.empty:
        logger.warning("DataFrame vazio ao tentar gerar o gráfico por UF.")
        return px.bar(title="Sem dados para exibir no momento")

    df_uf = (
        df.groupby("uf")
        .agg(
            total_clientes=("previsao", "count"),
            qtd_regularizam=("previsao", lambda x: (x == 1).sum()),
        )
        .reset_index()
    )

    df_uf["taxa_regularizacao"] = (
        df_uf["qtd_regularizam"] / df_uf["total_clientes"]
    ) * 100
    df_uf = df_uf.sort_values(by="qtd_regularizam", ascending=True)

    fig = px.bar(
        df_uf,
        x="qtd_regularizam",
        y="uf",
        orientation="h",
        text_auto=True,
        title="<b>Volume de Clientes com Propensão à Regularização por UF</b>",
        labels={
            "qtd_regularizam": "Qtd. Clientes (Propensão = 1)",
            "uf": "Estado (UF)",
        },
        color="taxa_regularizacao",
        color_continuous_scale="Blues",
    )

    fig.update_layout(
        template="plotly_white",
        coloraxis_colorbar=dict(title="Taxa (%)"),
        xaxis=dict(showgrid=True),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    logger.success("Gráfico por UF gerado com sucesso.")
    return fig


def gerar_grafico_canal_preferencial(df: pd.DataFrame) -> go.Figure:
    """Gráfico 5: Distribuição da Propensão à Regularização por Canal Preferencial (Donut Chart)."""
    logger.info("Gerando gráfico por canal preferencial...")
    if df.empty:
        logger.warning("DataFrame vazio ao tentar gerar o gráfico por Canal Preferencial.")
        return px.pie(title="Sem dados para exibir no momento")

    # Filtra apenas os clientes com propensão positiva (previsao = 1)
    df_propensos = df[df["previsao"] == 1]

    if df_propensos.empty:
        logger.warning("Nenhum cliente com previsão positiva (previsao = 1) para o gráfico de canais.")
        return px.pie(title="Nenhum cliente propenso encontrado no filtro atual")

    df_canal = (
        df_propensos["canal_preferencial"]
        .value_counts()
        .reset_index()
    )
    df_canal.columns = ["canal_preferencial", "quantidade"]

    fig = px.pie(
        df_canal,
        names="canal_preferencial",
        values="quantidade",
        hole=0.4,
        title="<b>Distribuição de Clientes Recuperáveis por Canal Preferencial</b>",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>Canal:</b> %{label}<br><b>Quantidade:</b> %{value}<br><b>Proporção:</b> %{percent}",
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    logger.success("Gráfico por canal preferencial gerado com sucesso.")
    return fig