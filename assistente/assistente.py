from loguru import logger
from langchain_core.messages import HumanMessage, ToolMessage


def executar_agente(pergunta, llm, tools):

    try:

        logger.info(
            f"Executando agente para pergunta: {pergunta}"
        )

        llm_com_tools = llm.bind_tools(tools)

        # ============================================================
        # PRIMEIRA CHAMADA DA LLM
        # ============================================================

        mensagem = llm_com_tools.invoke(
            [
                HumanMessage(
                    content=pergunta
                )
            ]
        )

        logger.info(
            f"Primeira resposta recebida. "
            f"Tool calls: {len(mensagem.tool_calls)}"
        )

        # ============================================================
        # CASO A LLM NÃO SOLICITE TOOL
        # ============================================================

        if not mensagem.tool_calls:

            logger.info(
                "LLM respondeu sem utilizar ferramentas."
            )

            return mensagem

        # ============================================================
        # MAPA DE TOOLS
        # ============================================================

        tools_por_nome = {
            tool.name: tool
            for tool in tools
        }

        # ============================================================
        # HISTÓRICO DA CONVERSA
        # ============================================================

        mensagens = [
            HumanMessage(
                content=pergunta
            ),
            mensagem
        ]

        # ============================================================
        # EXECUÇÃO DAS TOOLS
        # ============================================================

        for tool_call in mensagem.tool_calls:

            nome_tool = tool_call["name"]
            argumentos = tool_call["args"]
            tool_call_id = tool_call["id"]

            logger.info(
                f"Executando tool: {nome_tool}"
            )

            tool = tools_por_nome.get(
                nome_tool
            )

            if tool is None:

                raise ValueError(
                    f"Tool '{nome_tool}' não encontrada."
                )

            resultado = tool.invoke(
                argumentos
            )

            logger.info(
                f"Tool {nome_tool} executada com sucesso."
            )

            mensagens.append(
                ToolMessage(
                    content=str(resultado),
                    tool_call_id=tool_call_id
                )
            )

        logger.info(
            "Tools executadas. "
            "Enviando resultado para LLM."
        )

        # ============================================================
        # SEGUNDA CHAMADA DA LLM
        # ============================================================

        resposta_final = llm_com_tools.invoke(
            mensagens
        )

        logger.info(
            "Resposta final gerada com sucesso."
        )

        return resposta_final

    except Exception as excecao:

        logger.exception(
            f"Erro ao executar agente: {excecao}"
        )

        raise