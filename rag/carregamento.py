# Métodos que carregam todos os embeddings, coleções, banco vetorial ChromaDB que será usado pela RAG
import chromadb
from sentence_transformers import SentenceTransformer
from loguru import logger
from pathlib import Path
# ============================================================
# CARREGAMENTO DO CLIENTE DO BANCO E COLEÇÕES
# ============================================================

def carregar_colecao_rag(
    caminho_db,
    nome_colecao
):
    """
    Carrega uma coleção existente do ChromaDB.

    Parameters
    ----------
    caminho_db : str
        Caminho onde o ChromaDB está persistido.

    nome_colecao : str
        Nome da coleção que será carregada.

    Returns
    -------
    client : chromadb.PersistentClient
        Cliente do ChromaDB.

    collection : chromadb.Collection
        Coleção carregada.

    Raises
    ------
    Exception
        Caso ocorra algum erro ao inicializar o ChromaDB
        ou carregar a coleção.
    """
    try:

        logger.info(
            f"Iniciando carregamento da coleção "
            f"'{nome_colecao}'."
        )

        client = chromadb.PersistentClient(
            path=caminho_db
        )

        collection = client.get_collection(
            name=nome_colecao
        )

        quantidade_documentos = collection.count()

        logger.info(
            f"Coleção '{nome_colecao}' carregada "
            f"com sucesso."
        )

        logger.info(
            f"Quantidade de documentos: "
            f"{quantidade_documentos}"
        )

        return client, collection

    except Exception as erro:

        logger.exception(
            f"Erro ao carregar a coleção "
            f"'{nome_colecao}' do ChromaDB: {erro}"
        )

        raise


# ============================================================
# MODELO DE EMBEDDINGS
# ============================================================
def carregar_modelo_embedding(
    nome_modelo=(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
):
    """
    Carrega o modelo de embeddings.

    Parameters
    ----------
    nome_modelo : str
        Nome do modelo SentenceTransformer.

    Returns
    -------
    SentenceTransformer
        Modelo de embeddings carregado.

    Raises
    ------
    Exception
        Caso ocorra algum erro durante o carregamento.
    """

    try:

        logger.info(
            f"Iniciando carregamento do modelo "
            f"de embeddings: '{nome_modelo}'."
        )

        modelo = SentenceTransformer(
            nome_modelo
        )

        logger.info(
            f"Modelo de embeddings "
            f"'{nome_modelo}' carregado com sucesso."
        )

        return modelo

    except Exception as erro:

        logger.exception(
            f"Erro ao carregar o modelo de embeddings "
            f"'{nome_modelo}': {erro}"
        )

        raise

# Obs: Embora não seja um função explícita de carregamento, ela é responsável por vetorizar um dataframe
# padronizado de texto em um banco vetorial criando também a respectiva coleção..
# ============================================================
# CRIAÇÃO DO CHROMADB
# ============================================================
def criar_chromadb(
    df,
    modelo_embedding,
    nome_colecao="documentos"
):
    """
    Cria/sobrescreve uma coleção no ChromaDB.

    A base vetorial será criada na pasta 'chroma_db',
    localizada na raiz do projeto, no mesmo nível do app.py.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contendo as colunas 'id' e 'texto'.

    modelo_embedding : SentenceTransformer
        Modelo utilizado para gerar os embeddings.

    nome_colecao : str
        Nome da coleção no ChromaDB.

    Returns
    -------
    client : chromadb.PersistentClient
        Cliente do ChromaDB.

    collection : chromadb.Collection
        Coleção criada.

    Raises
    ------
    Exception
        Caso ocorra algum erro durante a criação da base vetorial.
    """

    try:

        # --------------------------------------------------------
        # Caminho da base vetorial
        # --------------------------------------------------------

        caminho_projeto = Path(__file__).resolve().parent.parent

        caminho_db = caminho_projeto / "chroma_db"

        caminho_db.mkdir(
            parents=True,
            exist_ok=True
        )

        logger.info(
            f"Diretório do ChromaDB: {caminho_db}"
        )

        # --------------------------------------------------------
        # Validação do DataFrame
        # --------------------------------------------------------

        colunas_obrigatorias = {
            "id",
            "texto"
        }

        colunas_ausentes = (
            colunas_obrigatorias
            - set(df.columns)
        )

        if colunas_ausentes:

            raise ValueError(
                "DataFrame não possui as colunas obrigatórias: "
                f"{sorted(colunas_ausentes)}"
            )

        if df.empty:

            raise ValueError(
                "O DataFrame está vazio."
            )

        # --------------------------------------------------------
        # Cliente ChromaDB
        # --------------------------------------------------------

        logger.info(
            "Inicializando cliente ChromaDB."
        )

        cliente_chroma = chromadb.PersistentClient(
            path=str(caminho_db)
        )

        # --------------------------------------------------------
        # Sobrescrever coleção existente
        # --------------------------------------------------------

        colecoes_existentes = [
            colecao.name
            for colecao in cliente_chroma.list_collections()
        ]

        if nome_colecao in colecoes_existentes:

            logger.warning(
                f"Coleção '{nome_colecao}' já existe. "
                "A coleção será sobrescrita."
            )

            cliente_chroma.delete_collection(
                name=nome_colecao
            )

            logger.info(
                f"Coleção '{nome_colecao}' anterior "
                "removida com sucesso."
            )

        # --------------------------------------------------------
        # Criar nova coleção
        # --------------------------------------------------------

        colecao = cliente_chroma.create_collection(
            name=nome_colecao
        )

        logger.info(
            f"Coleção '{nome_colecao}' criada com sucesso."
        )

        # --------------------------------------------------------
        # Preparar documentos
        # --------------------------------------------------------

        documentos = (
            df["texto"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        ids = (
            df["id"]
            .astype(str)
            .tolist()
        )

        logger.info(
            f"Quantidade de documentos para vetorização: "
            f"{len(documentos)}"
        )

        # --------------------------------------------------------
        # Gerar embeddings
        # --------------------------------------------------------

        logger.info(
            "Iniciando geração dos embeddings."
        )

        embeddings = modelo_embedding.encode(
            documentos,
            show_progress_bar=True
        )

        logger.info(
            "Embeddings gerados com sucesso."
        )

        # --------------------------------------------------------
        # Inserir documentos na coleção
        # --------------------------------------------------------

        colecao.add(
            ids=ids,
            documents=documentos,
            embeddings=embeddings.tolist()
        )

        logger.info(
            f"{len(documentos)} documentos adicionados "
            f"à coleção '{nome_colecao}'."
        )

        # --------------------------------------------------------
        # Retorno
        # --------------------------------------------------------

        return cliente_chroma, colecao

    except Exception as erro:

        logger.exception(
            f"Erro ao criar/sobrescrever a coleção "
            f"'{nome_colecao}': {erro}"
        )

        raise
