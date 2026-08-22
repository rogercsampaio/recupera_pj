from pathlib import Path
import pandas as pd
from loguru import logger
from langchain_core.tools import tool
from rag.busca import buscar_documentos

from rag.carregamento import (
    carregar_colecao_rag,
    carregar_modelo_embedding
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

RAIZ_PROJETO = Path(__file__).resolve().parent.parent

CAMINHO_CLIENTES = (
    RAIZ_PROJETO
    / "bases_tratadas"
    / "clientes_sem_nulos.csv"
)

CAMINHO_INTERACOES = (
    RAIZ_PROJETO
    / "bases_tratadas"
    / "interacoes_com_info_data.csv"
)

CAMINHO_PREVISOES = (
    RAIZ_PROJETO
    / "resultados"
    / "xgboost_inferencia_todos.csv"
)

CAMINHO_CHROMADB = (
    RAIZ_PROJETO
    / "chroma_db"
)

NOME_COLECAO_RAG = "documentos"

@tool
def recomendar_acoes(id_cliente: str) -> dict:
    """
    Gera uma recomendação explicável de ação para um cliente PJ.

    ```
    A recomendação combina:
    - perfil cadastral, financeiro e de relacionamento;
    - previsão de regularização;
    - histórico e resultado das interações;
    - regras e restrições presentes na documentação da RAG.

    A recomendação informa:
    - ação proposta;
    - justificativa;
    - evidências utilizadas;
    - restrições aplicáveis;
    - nível de confiança;
    - condição de encaminhamento para análise humana.

    O identificador deve ser informado no formato PJXXXX,
    por exemplo: PJ0005.
    """
    try:
        logger.info(
            f"Iniciando recomendação de ações para "
            f"o cliente: {id_cliente}"
        )

        # ========================================================
        # 1. PERFIL DO CLIENTE
        # ========================================================

        cliente = consultar_cliente.invoke({
            "id_cliente": id_cliente
        })

        if not cliente.get("sucesso"):
            return cliente

        dados_cliente = cliente.get("dados", {})

        # ========================================================
        # 2. SCORE DE REGULARIZAÇÃO
        # ========================================================

        score = consultar_score_regularizacao.invoke({
            "id_cliente": id_cliente
        })

        if not score.get("sucesso"):
            return score

        dados_score = score.get(
            "dados",
            score
        )

        probabilidade = dados_score.get(
            "prob_regularizacao"
        )

        previsao = dados_score.get(
            "previsao"
        )

        data_inicio = dados_score.get(
            "data_inicio_horizonte"
        )

        data_fim = dados_score.get(
            "data_fim_horizonte"
        )

        # ========================================================
        # 3. HISTÓRICO DE INTERAÇÕES
        # ========================================================

        historico = buscar_historico_interacoes.invoke({
            "id_cliente": id_cliente
        })

        if not historico.get("sucesso"):
            return historico

        interacoes = historico.get(
            "interacoes",
            []
        )

        # ========================================================
        # 4. DOCUMENTAÇÃO / REGRAS DE NEGÓCIO
        # ========================================================

        consulta_documentacao = (
            "Quais regras, políticas, restrições e procedimentos "
            "vigentes devem ser considerados para definir ações "
            "de recuperação e regularização de clientes PJ, "
            "incluindo contato, negociação, parcelamento, ofertas, "
            "canais de comunicação, limites de atuação e situações "
            "que exigem análise humana?"
        )

        documentacao = buscar_documentacao.invoke({
            "consulta": consulta_documentacao,
            "n_resultados": 5
        })

        if not documentacao.get("sucesso"):
            return documentacao

        restricoes = documentacao.get(
            "resultados",
            []
        )

        # ========================================================
        # 5. CLASSIFICAR PROBABILIDADE
        # ========================================================

        if probabilidade is not None:
            if probabilidade >= 0.70:
                faixa_probabilidade = "alta"
            elif probabilidade >= 0.50:
                faixa_probabilidade = "media"
            else:
                faixa_probabilidade = "baixa"
        else:
            faixa_probabilidade = "indisponivel"

        # ========================================================
        # 6. ANALISAR HISTÓRICO
        # ========================================================

        quantidade_interacoes = len(interacoes)

        resultados_interacoes = [
            interacao.get("resultado")
            for interacao in interacoes
        ]

        canais_interacoes = [
            interacao.get("canal")
            for interacao in interacoes
        ]

        datas = [
            interacao.get("data_interacao")
            for interacao in interacoes
            if interacao.get("data_interacao")
        ]

        ultima_interacao = (
            max(datas)
            if datas
            else None
        )

        # ========================================================
        # 7. IDENTIFICAR SINAIS DO HISTÓRICO
        # ========================================================

        teve_promessa = (
            "promessa_registrada"
            in resultados_interacoes
        )

        teve_recusa = (
            "recusa_oferta"
            in resultados_interacoes
        )

        teve_intencao_pagamento = (
            "intencao_pagamento"
            in resultados_interacoes
        )

        teve_dificuldade = (
            "dificuldade_temporaria"
            in resultados_interacoes
        )

        teve_contestacao = (
            "contestacao"
            in resultados_interacoes
        )

        teve_recuperacao_judicial = (
            "recuperacao_judicial"
            in resultados_interacoes
        )

        # ========================================================
        # 8. DEFINIR AÇÃO BASE
        # ========================================================

        acao = "priorizacao_operacional"

        justificativa = (
            "Não foram identificadas evidências suficientes "
            "para recomendar uma ação específica."
        )

        confianca = "baixa"

        # ========================================================
        # 8.1 RECUPERAÇÃO JUDICIAL
        # ========================================================

        if teve_recuperacao_judicial:
            acao = "analise_especializada"

            justificativa = (
                "O histórico registra recuperação judicial. "
                "A situação requer análise específica antes "
                "de qualquer ação de recuperação."
            )

            confianca = "alta"

        # ========================================================
        # 8.2 CONTESTAÇÃO
        # ========================================================

        elif teve_contestacao:
            acao = "analise_especializada"

            justificativa = (
                "O histórico registra contestação. "
                "A situação deve ser analisada antes "
                "de uma nova ação de recuperação."
            )

            confianca = "alta"

        # ========================================================
        # 8.3 PROMESSA DE PAGAMENTO
        # ========================================================

        elif teve_promessa:
            acao = "nova_tentativa_de_contato"

            justificativa = (
                "O histórico registra promessa de pagamento, "
                "indicando intenção prévia de regularização. "
                "Recomenda-se acompanhamento da promessa, "
                "respeitando as regras vigentes."
            )

            confianca = "alta"

        # ========================================================
        # 8.4 DIFICULDADE TEMPORÁRIA
        # ========================================================

        elif teve_dificuldade:
            acao = "analise_especializada"

            justificativa = (
                "O histórico registra dificuldade temporária. "
                "A situação deve ser analisada considerando "
                "as alternativas permitidas pela política vigente."
            )

            confianca = "media"

        # ========================================================
        # 8.5 RECUSA DE OFERTA
        # ========================================================

        elif teve_recusa:
            acao = "nova_tentativa_de_contato"

            justificativa = (
                "O histórico registra recusa de oferta. "
                "Recomenda-se evitar a repetição imediata da "
                "mesma abordagem e considerar canal ou momento "
                "alternativo, quando permitido."
            )

            confianca = "media"

        # ========================================================
        # 8.6 INTENÇÃO DE PAGAMENTO + SCORE
        # ========================================================

        elif teve_intencao_pagamento:

            if faixa_probabilidade == "alta":
                acao = "contato_digital"

                justificativa = (
                    "O histórico registra intenção de pagamento "
                    "e o modelo indica alta probabilidade de "
                    "regularização no horizonte previsto."
                )

                confianca = "alta"

            elif faixa_probabilidade == "media":
                acao = "contato_digital"

                justificativa = (
                    "O histórico registra intenção de pagamento "
                    "e o modelo indica probabilidade intermediária "
                    "de regularização no horizonte previsto."
                )

                confianca = "media"

            else:
                acao = "analise_especializada"

                justificativa = (
                    "Embora exista intenção de pagamento registrada, "
                    "a probabilidade estimada de regularização é baixa. "
                    "Recomenda-se avaliar a estratégia de abordagem."
                )

                confianca = "media"

        # ========================================================
        # 8.7 AUSÊNCIA DE SINAIS ESPECÍFICOS + SCORE
        # ========================================================

        else:

            if faixa_probabilidade == "alta":
                acao = "contato_digital"

                justificativa = (
                    "Não foram identificados resultados específicos "
                    "no histórico, mas o modelo indica alta "
                    "probabilidade de regularização no horizonte previsto."
                )

                confianca = "media"

            elif faixa_probabilidade == "media":
                acao = "priorizacao_operacional"

                justificativa = (
                    "Não foram identificados sinais específicos "
                    "no histórico e a probabilidade estimada "
                    "de regularização é intermediária."
                )

                confianca = "media"

            elif faixa_probabilidade == "baixa":
                acao = "priorizacao_operacional"

                justificativa = (
                    "Não foram identificados sinais positivos "
                    "no histórico e a probabilidade estimada "
                    "de regularização é baixa."
                )

                confianca = "media"

            else:
                acao = "analise_especializada"

                justificativa = (
                    "Não foi possível obter a probabilidade "
                    "de regularização para apoiar a decisão."
                )

                confianca = "baixa"

        # ========================================================
        # 9. CONDIÇÃO DE ANÁLISE HUMANA
        # ========================================================

        encaminhamento_humano = False
        motivo_encaminhamento = None

        if teve_recuperacao_judicial:
            encaminhamento_humano = True

            motivo_encaminhamento = (
                "Cliente possui registro de recuperação judicial."
            )

        elif teve_contestacao:
            encaminhamento_humano = True

            motivo_encaminhamento = (
                "Cliente possui registro de contestação."
            )

        elif teve_dificuldade:
            encaminhamento_humano = True

            motivo_encaminhamento = (
                "Cliente possui registro de dificuldade temporária "
                "e a ação depende das alternativas permitidas."
            )

        elif probabilidade is None:
            encaminhamento_humano = True

            motivo_encaminhamento = (
                "Não foi possível obter a probabilidade "
                "de regularização."
            )

        # ========================================================
        # 10. REGRAS DA RAG
        # ========================================================

        texto_restricoes = " ".join(
            str(restricao)
            for restricao in restricoes
        ).lower()

        if (
            "análise humana" in texto_restricoes
            or "analise humana" in texto_restricoes
            or "análise especializada" in texto_restricoes
            or "analise especializada" in texto_restricoes
        ):
            encaminhamento_humano = True

            if motivo_encaminhamento is None:
                motivo_encaminhamento = (
                    "As regras recuperadas da documentação "
                    "indicam necessidade de análise humana "
                    "ou especializada."
                )

        # ========================================================
        # 11. EVIDÊNCIAS UTILIZADAS
        # ========================================================

        evidencias = {
            "perfil_cliente": dados_cliente,
            "previsao_regularizacao": previsao,
            "probabilidade_regularizacao": probabilidade,
            "faixa_probabilidade": faixa_probabilidade,
            "data_inicio_horizonte": data_inicio,
            "data_fim_horizonte": data_fim,
            "quantidade_interacoes": quantidade_interacoes,
            "ultima_interacao": ultima_interacao,
            "canais_utilizados": list(
                set(canais_interacoes)
            ),
            "resultados_interacoes": resultados_interacoes
        }

        # ========================================================
        # 12. RETORNO DA RECOMENDAÇÃO
        # ========================================================

        recomendacao = {
            "acao_proposta": acao,
            "justificativa": justificativa,
            "evidencias_utilizadas": evidencias,
            "restricoes_aplicaveis": restricoes,
            "nivel_confianca": confianca,
            "encaminhamento_analise_humana": (
                encaminhamento_humano
            ),
            "motivo_encaminhamento_humano": (
                motivo_encaminhamento
            )
        }

        logger.info(
            f"Recomendação concluída para {id_cliente}. "
            f"Ação: {acao} | "
            f"Probabilidade: {probabilidade} | "
            f"Confiança: {confianca} | "
            f"Análise humana: {encaminhamento_humano}"
        )

        return {
            "sucesso": True,
            "id_cliente": id_cliente,
            "recomendacao": recomendacao
        }
    except Exception as excecao:
        logger.exception(
            f"Erro ao gerar recomendação para "
            f"o cliente {id_cliente}: {excecao}"
        )

        return {
            "sucesso": False,
            "id_cliente": id_cliente,
            "erro": str(excecao)
        }


# ============================================================
# CARREGAMENTO DAS PREVISÕES DE TODOS OS CLIENTES. 
# Obs: Inclui os clientes que foram usados no treino e teste.
# ============================================================
def carregar_previsoes_todos_clientes():
    try:
        logger.info("Carregando as previsões (todos os clientes).")
        df = pd.read_csv(CAMINHO_PREVISOES)
        logger.info(
            f"Base de previsões carregada com sucesso"
            f"Registros: {len(df)}")
        return df
    except Exception as excecao:
        logger.exception(f"Erro ao carregar base de previsoes: {excecao}")
        raise

@tool
def consultar_score_regularizacao(id_cliente: str) -> dict:
    """
    Consulta a previsão de regularização de um cliente PJ.

    Retorna a previsão do modelo, a probabilidade estimada de
    regularização e o horizonte temporal correspondente à
    data de referência.

    A `data_referencia` representa a data-base utilizada pelo
    modelo para realizar a previsão.

    As informações do cliente consideradas pelo modelo são
    referentes a essa data. A previsão se refere ao período
    futuro definido por `data_inicio_horizonte` e
    `data_fim_horizonte`.

    Exemplo:
    Se `data_referencia` for 2026-02-28,
    `data_inicio_horizonte` for 2026-03-01 e
    `data_fim_horizonte` for 2026-03-30, a previsão representa
    a probabilidade estimada de regularização do cliente entre
    01/03/2026 e 30/03/2026.

    Use esta ferramenta quando precisar consultar a previsão,
    a probabilidade de regularização ou o horizonte temporal
    da previsão de um cliente específico.

    O identificador deve ser informado no formato `PJXXXX`,
    por exemplo: `PJ0005`.
    """
    try:
        logger.info(
            f"Consultando score de regularização: {id_cliente}"
        )

        df = carregar_previsoes_todos_clientes()

        resultado = df[
            df["id_cliente"].astype(str) == str(id_cliente)
        ]

        if resultado.empty:
            logger.warning(
                f"Previsão não encontrada para o cliente: {id_cliente}"
            )
            return {
                "sucesso": False,
                "id_cliente": id_cliente,
                "erro": "Previsão de regularização não encontrada."
            }

        dados = resultado.iloc[0].to_dict()

        logger.info(
            f"Score de regularização do cliente "
            f"{id_cliente} encontrado com sucesso."
        )

        return {
            "sucesso": True,
            "id_cliente": id_cliente,
            "data_referencia": dados.get("data_referencia"),
            "data_inicio_horizonte": dados.get(
                "data_inicio_horizonte"
            ),
            "data_fim_horizonte": dados.get(
                "data_fim_horizonte"
            ),
            "previsao": dados.get("previsao"),
            "prob_regularizacao": dados.get(
                "prob_regularizacao"
            )
        }

    except Exception as excecao:
        logger.exception(
            f"Erro ao consultar score de regularização "
            f"{id_cliente}: {excecao}"
        )
        return {
            "sucesso": False,
            "id_cliente": id_cliente,
            "erro": str(excecao)
        }


# ============================================================
# CARREGAMENTO DOS CLIENTES
# ============================================================
def carregar_clientes():

    try:

        logger.info(
            "Carregando base estruturada de clientes."
        )

        df = pd.read_csv(
            CAMINHO_CLIENTES
        )

        logger.info(
            f"Base de clientes carregada com sucesso. "
            f"Registros: {len(df)}"
        )

        return df

    except Exception as excecao:

        logger.exception(
            f"Erro ao carregar base de clientes: {excecao}"
        )

        raise


# ============================================================
# CARREGAMENTO DAS INTERAÇÕES
# ============================================================
def carregar_interacoes():

    try:

        logger.info(
            "Carregando base de histórico de interações."
        )

        df = pd.read_csv(
            CAMINHO_INTERACOES
        )

        logger.info(
            f"Base de interações carregada com sucesso. "
            f"Registros: {len(df)}"
        )

        return df

    except Exception as excecao:

        logger.exception(
            f"Erro ao carregar base de interações: {excecao}"
        )

        raise


# ============================================================
# TOOL — CONSULTAR CLIENTE
# ============================================================

@tool
def consultar_cliente(id_cliente: str) -> dict:

    """
    Consulta os dados estruturados de um cliente PJ
    pelo seu identificador único.

    Use esta ferramenta quando precisar consultar
    informações cadastrais, financeiras, de crédito,
    relacionamento ou comportamento do cliente.

    O identificador deve ser informado no formato
    PJXXXX, por exemplo: PJ0005.
    """

    try:

        logger.info(
            f"Consultando cliente: {id_cliente}"
        )

        df = carregar_clientes()

        resultado = df[
            df["id_cliente"].astype(str) == str(id_cliente)
        ]

        if resultado.empty:

            logger.warning(
                f"Cliente não encontrado: {id_cliente}"
            )

            return {
                "sucesso": False,
                "id_cliente": id_cliente,
                "erro": "Cliente não encontrado."
            }

        dados = resultado.iloc[0].to_dict()

        logger.info(
            f"Cliente {id_cliente} encontrado com sucesso."
        )

        return {
            "sucesso": True,
            "id_cliente": id_cliente,
            "dados": dados
        }

    except Exception as excecao:

        logger.exception(
            f"Erro ao consultar cliente "
            f"{id_cliente}: {excecao}"
        )

        return {
            "sucesso": False,
            "id_cliente": id_cliente,
            "erro": str(excecao)
        }


# ============================================================
# TOOL — BUSCAR HISTÓRICO DE INTERAÇÕES
# ============================================================
@tool
def buscar_historico_interacoes(
    id_cliente: str
) -> dict:

    """
    Consulta o histórico de interações de um cliente PJ.

    Use esta ferramenta quando o usuário perguntar sobre
    contatos, interações, atendimentos, histórico de cobrança
    ou qualquer registro de interação relacionado a um cliente.

    O identificador deve ser informado no formato PJXXXX,
    por exemplo: PJ0005.

    Parameters
    ----------
    id_cliente : str
        Identificador único do cliente.
    """

    try:

        logger.info(
            f"Consultando histórico de interações: "
            f"{id_cliente}"
        )

        df = carregar_interacoes()

        if "id_cliente" not in df.columns:

            raise ValueError(
                "A base de interações não possui a coluna "
                "'id_cliente'."
            )

        resultado = df[
            df["id_cliente"].astype(str) == str(id_cliente)
        ]

        if resultado.empty:

            logger.warning(
                f"Nenhuma interação encontrada para "
                f"o cliente: {id_cliente}"
            )

            return {
                "sucesso": True,
                "id_cliente": id_cliente,
                "quantidade_interacoes": 0,
                "interacoes": []
            }

        interacoes = (
            resultado
            .where(pd.notna(resultado), None)
            .to_dict(orient="records")
        )

        logger.info(
            f"Histórico encontrado para {id_cliente}. "
            f"Quantidade de interações: {len(interacoes)}"
        )

        return {
            "sucesso": True,
            "id_cliente": id_cliente,
            "quantidade_interacoes": len(interacoes),
            "interacoes": interacoes
        }

    except Exception as excecao:

        logger.exception(
            f"Erro ao consultar histórico de interações "
            f"do cliente {id_cliente}: {excecao}"
        )

        return {
            "sucesso": False,
            "id_cliente": id_cliente,
            "erro": str(excecao)
        }


# ============================================================
# TOOL — BUSCAR DOCUMENTAÇÃO
# ============================================================
@tool
def buscar_documentacao(
    consulta: str,
    n_resultados: int = 5
) -> dict:

    """
    Realiza uma busca semântica na documentação
    armazenada na base vetorial da RAG.

    Use esta ferramenta quando o usuário perguntar
    sobre políticas, regras, procedimentos, conceitos,
    normas ou qualquer informação que esteja presente
    na documentação indexada.

    A busca utiliza embeddings e ChromaDB para recuperar
    os documentos semanticamente mais relevantes.

    Parameters
    ----------
    consulta : str
        Pergunta ou assunto que deve ser pesquisado
        na documentação.

    n_resultados : int
        Quantidade de documentos relevantes retornados.
    """

    try:

        logger.info(
            f"Iniciando busca na documentação: "
            f"'{consulta}'"
        )

        # --------------------------------------------------------
        # Carregar modelo de embeddings
        # --------------------------------------------------------

        modelo_embedding = carregar_modelo_embedding()

        # --------------------------------------------------------
        # Carregar coleção do ChromaDB
        # --------------------------------------------------------

        _, colecao = carregar_colecao_rag(
            caminho_db=str(CAMINHO_CHROMADB),
            nome_colecao=NOME_COLECAO_RAG
        )

        # --------------------------------------------------------
        # Busca semântica
        # --------------------------------------------------------

        resultados = buscar_documentos(
            consulta=consulta,
            modelo_embedding=modelo_embedding,
            colecao=colecao,
            n_resultados=n_resultados
        )

        logger.info(
            f"Busca na documentação concluída. "
            f"{len(resultados)} documentos encontrados."
        )

        return {
            "sucesso": True,
            "consulta": consulta,
            "resultados": resultados
        }

    except Exception as excecao:

        logger.exception(
            f"Erro ao buscar documentação: {excecao}"
        )

        return {
            "sucesso": False,
            "consulta": consulta,
            "erro": str(excecao)
        }