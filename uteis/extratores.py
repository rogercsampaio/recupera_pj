import pandas as pd
from pathlib import Path
from loguru import logger
import requests

# ============================================================
#  Função de que ler um arquivo csv
# ============================================================
def ler_base_por_csv(caminho_arquivo):
    """
    Lê um arquivo CSV e retorna um DataFrame do Pandas.
    """
    try:
        base_dataframe = pd.read_csv(caminho_arquivo)
        logger.info(f"Arquivo lido com sucesso")
        return base_dataframe
    except Exception as excecao:
        logger.error(f"Erro ao ler o arquivo: {excecao}")
        return None


# ============================================================
# CARREGAMENTO DOS DOCUMENTOS TXT
# ============================================================
def carregar_txts(pasta):
    """
    Carrega arquivos TXT de uma pasta e transforma
    seu conteúdo em um DataFrame.

    Parameters
    ----------
    pasta : str ou Path
        Caminho da pasta contendo os arquivos TXT.

    Returns
    -------
    pandas.DataFrame
        DataFrame contendo:
        - nome_arquivo
        - texto

    Raises
    ------
    FileNotFoundError
        Caso a pasta informada não exista.

    ValueError
        Caso nenhum arquivo TXT seja encontrado.

    Exception
        Caso ocorra algum erro durante o processamento.
    """

    try:

        # --------------------------------------------------------
        # Validar pasta
        # --------------------------------------------------------

        pasta = Path(pasta)

        logger.info(
            f"Iniciando carregamento dos arquivos TXT "
            f"da pasta: {pasta}"
        )

        if not pasta.exists():

            raise FileNotFoundError(
                f"A pasta '{pasta}' não existe."
            )

        if not pasta.is_dir():

            raise NotADirectoryError(
                f"O caminho '{pasta}' não é uma pasta."
            )

        # --------------------------------------------------------
        # Localizar arquivos TXT
        # --------------------------------------------------------

        arquivos = sorted(
            pasta.glob("*.txt")
        )

        if not arquivos:

            raise ValueError(
                f"Nenhum arquivo TXT encontrado "
                f"na pasta '{pasta}'."
            )

        logger.info(
            f"{len(arquivos)} arquivos TXT encontrados."
        )

        # --------------------------------------------------------
        # Ler documentos
        # --------------------------------------------------------

        documentos = []

        for arquivo in arquivos:

            logger.debug(
                f"Lendo arquivo: {arquivo.name}"
            )

            with open(
                arquivo,
                "r",
                encoding="utf-8"
            ) as f:

                texto = f.read()

            documentos.append(
                {
                    "nome_arquivo": arquivo.name,
                    "texto": texto
                }
            )

        # --------------------------------------------------------
        # DataFrame
        # --------------------------------------------------------

        df = pd.DataFrame(
            documentos
        )

        logger.info(
            f"Processo de extração de textos concluído. "
            f"{len(df)} documentos carregados."
        )

        return df

    except Exception as excecao:

        logger.exception(
            f"Houve um erro ao carregar e processar "
            f"os arquivos para o RAG: {excecao}"
        )

        raise
    
# =====================================================================
# COTAÇÃO USD / BRL
# =====================================================================
def get_usd_brl():
    """
    Busca a cotação atual do dólar em relação ao real.
    """

    url = "https://economia.awesomeapi.com.br/json/last/USD-BRL"

    try:
        response = requests.get(
            url,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        price = float(
            data["USDBRL"]["bid"]
        )

        logger.debug(
            f"Cotação USD/BRL obtida com sucesso: {price:.4f}"
        )

        return price

    except Exception:
        logger.exception(
            "Erro ao buscar cotação USD/BRL."
        )

        return None
