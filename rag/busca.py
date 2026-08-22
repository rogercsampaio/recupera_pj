from loguru import logger
import numpy as np

# ============================================================
# 3. SIMILARIDADE DE COSSENO
# ============================================================

def similaridade_cosseno(
    vetor_a,
    vetor_b
):
    """
    Calcula a similaridade de cosseno entre dois vetores.

    Parameters
    ----------
    vetor_a : array-like
        Primeiro vetor.

    vetor_b : array-like
        Segundo vetor.

    Returns
    -------
    float
        Similaridade de cosseno entre os vetores.
    """

    try:

        # --------------------------------------------------------
        # Converter para numpy array
        # --------------------------------------------------------

        vetor_a = np.array(
            vetor_a
        )

        vetor_b = np.array(
            vetor_b
        )

        # --------------------------------------------------------
        # Calcular normas
        # --------------------------------------------------------

        norma_a = np.linalg.norm(
            vetor_a
        )

        norma_b = np.linalg.norm(
            vetor_b
        )

        # --------------------------------------------------------
        # Evitar divisão por zero
        # --------------------------------------------------------

        if norma_a == 0 or norma_b == 0:

            logger.warning(
                "Similaridade de cosseno não pôde ser calculada "
                "porque um dos vetores possui norma zero."
            )

            return 0.0

        # --------------------------------------------------------
        # Similaridade
        # --------------------------------------------------------

        similaridade = np.dot(
            vetor_a,
            vetor_b
        ) / (
            norma_a * norma_b
        )

        return float(
            similaridade
        )

    except Exception as excecao:

        logger.exception(
            f"Erro ao calcular similaridade de cosseno: "
            f"{excecao}"
        )

        raise


# ============================================================
# 4. BUSCA SEMÂNTICA
# ============================================================

def buscar_documentos(
    consulta,
    modelo_embedding,
    colecao,
    n_resultados=5
):
    """
    Busca documentos semanticamente similares.

    Parameters
    ----------
    consulta : str
        Texto utilizado para realizar a busca.

    modelo_embedding : SentenceTransformer
        Modelo utilizado para gerar o embedding da consulta.

    colecao : chromadb.Collection
        Coleção do ChromaDB.

    n_resultados : int
        Quantidade de documentos retornados.

    Returns
    -------
    list
        Lista contendo:

        - id_documento
        - texto
        - similaridade_cosseno
        - score_similaridade (0-100)
    """

    try:

        # --------------------------------------------------------
        # Validações
        # --------------------------------------------------------

        if not consulta or not str(consulta).strip():

            raise ValueError(
                "A consulta não pode ser vazia."
            )

        if n_resultados <= 0:

            raise ValueError(
                "n_resultados deve ser maior que zero."
            )

        logger.info(
            f"Iniciando busca semântica. "
            f"Consulta: '{consulta}' | "
            f"Top-K: {n_resultados}"
        )

        # --------------------------------------------------------
        # Embedding da consulta
        # --------------------------------------------------------

        logger.debug(
            "Gerando embedding da consulta."
        )

        embedding_consulta = (
            modelo_embedding.encode(
                consulta
            )
        )

        logger.debug(
            "Embedding da consulta gerado com sucesso."
        )

        # --------------------------------------------------------
        # Buscar documentos no ChromaDB
        # --------------------------------------------------------

        logger.debug(
            "Consultando coleção no ChromaDB."
        )

        resultado = colecao.query(
            query_embeddings=[
                embedding_consulta.tolist()
            ],
            n_results=n_resultados,
            include=[
                "documents",
                "embeddings"
            ]
        )

        # --------------------------------------------------------
        # Inicializar resultados
        # --------------------------------------------------------

        resultados = []

        ids_encontrados = resultado.get(
            "ids",
            [[]]
        )[0]

        logger.info(
            f"{len(ids_encontrados)} documentos "
            "recuperados do ChromaDB."
        )

        # --------------------------------------------------------
        # Calcular similaridade
        # --------------------------------------------------------

        for i in range(
            len(ids_encontrados)
        ):

            id_documento = resultado[
                "ids"
            ][0][i]

            documento = resultado[
                "documents"
            ][0][i]

            embedding_documento = resultado[
                "embeddings"
            ][0][i]

            # ----------------------------------------------------
            # Similaridade de cosseno
            # ----------------------------------------------------

            similaridade = similaridade_cosseno(
                embedding_consulta,
                embedding_documento
            )

            # ----------------------------------------------------
            # Converter para escala 0-100
            # ----------------------------------------------------

            score = (
                (similaridade + 1) / 2
            ) * 100

            resultados.append(
                {
                    "id_documento": id_documento,
                    "texto": documento,
                    "similaridade_cosseno": similaridade,
                    "score_similaridade": score
                }
            )

        # --------------------------------------------------------
        # Log dos resultados
        # --------------------------------------------------------

        logger.info(
            f"Busca semântica concluída. "
            f"{len(resultados)} documentos retornados."
        )

        for resultado_documento in resultados:

            logger.debug(
                f"Documento: "
                f"{resultado_documento['id_documento']} | "
                f"Similaridade: "
                f"{resultado_documento['score_similaridade']:.2f}%"
            )

        return resultados

    except Exception as excecao:

        logger.exception(
            f"Erro durante a busca semântica: "
            f"{excecao}"
        )

        raise