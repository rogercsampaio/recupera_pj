from datetime import datetime
import calendar
import streamlit as st
from modelagem_preditiva.uteis_modelagem import inferir_cliente_avulso

def render_modelo_preditivo_page():
  st.title("🔮 Modelo Preditivo")
  st.caption("Analytics & Scoring de Inadimplência")
  st.markdown("---")

  # Formulário principal
  with st.form(key="form_inferencia_cliente"):
    st.subheader("📋 Dados Financeiros e de Contrato")

    col1, col2, col3 = st.columns(3)

    with col1:
      dias_atraso = st.number_input(
          "Dias em Atraso",
          min_value=0,
          value=0,
          step=1,
          help="Quantidade de dias de atraso atual.",
      )

      qtd_parcelas_vencidas = st.number_input(
          "Qtd. Parcelas Vencidas", min_value=0, value=0, step=1
      )

    with col2:
      utilizacao_limite_pct = st.number_input(
          "Utilização do Limite (Escala 0 a 1+)",
          min_value=0.0,
          max_value=2.0,
          value=0.15,
          step=0.01,
          format="%.4f",
          help=(
              "Exemplo: 0.50 representa 50% de uso. Valores acima de 1.0"
              " indicam uso acima do limite contratado."
          ),
      )

      qtd_contratos_ativos = st.number_input(
          "Qtd. Contratos Ativos", min_value=0, value=1, step=1
      )

    with col3:
      dias_desde_ultima_interacao = st.number_input(
          "Dias Desde Última Interação",
          min_value=-1,
          value=-1,
          step=1,
          help="Informe -1 caso o cliente nunca tenha interagido.",
      )

    st.markdown("---")
    st.subheader("📞 Histórico de Interações (Últimos 30 dias)")

    col4, col5, col6 = st.columns(3)

    with col4:
      qtd_interacoes_ult_30d = st.number_input(
          "Total Interações (30d)", min_value=0, value=0, step=1
      )

      qtd_canais_distintos_ult_30d = st.number_input(
          "Qtd. Canais Distintos (30d)", min_value=0, value=0, step=1
      )

    with col5:
      qtd_interacoes_ativas_ult_30d = st.number_input(
          "Interações Ativas (30d)",
          min_value=0,
          value=0,
          step=1,
          help="Iniciadas pela empresa.",
      )

      qtd_interacoes_receptivas_ult_30d = st.number_input(
          "Interações Receptivas (30d)",
          min_value=0,
          value=0,
          step=1,
          help="Iniciadas pelo cliente.",
      )

    with col6:
      teve_promessa = st.selectbox(
          "Teve Promessa de Pagamento?",
          options=["Não", "Sim"],
          index=0,
          help="Indica se houve promessa de pagamento nos últimos 30 dias.",
      )
      teve_promessa_pagamento_ult_30d = 1.0 if teve_promessa == "Sim" else 0.0

      qtd_promessas_pagamento_ult_30d = st.number_input(
          "Qtd. Promessas de Pagamento", min_value=0, value=0, step=1
      )

    st.markdown("---")
    btn_submeter = st.form_submit_button(
        "🚀 Executar Inferência", use_container_width=True
    )

  # Processamento da inferência ao clicar no botão
  if btn_submeter:
    with st.spinner("Processando inferência com o modelo..."):
      try:
        # Execução da função do backend
        df_resultado = inferir_cliente_avulso(
            dias_atraso=dias_atraso,
            qtd_parcelas_vencidas=qtd_parcelas_vencidas,
            utilizacao_limite_pct=utilizacao_limite_pct,
            qtd_contratos_ativos=qtd_contratos_ativos,
            qtd_interacoes_ult_30d=qtd_interacoes_ult_30d,
            dias_desde_ultima_interacao=dias_desde_ultima_interacao,
            qtd_canais_distintos_ult_30d=qtd_canais_distintos_ult_30d,
            qtd_interacoes_ativas_ult_30d=qtd_interacoes_ativas_ult_30d,
            qtd_interacoes_receptivas_ult_30d=qtd_interacoes_receptivas_ult_30d,
            teve_promessa_pagamento_ult_30d=teve_promessa_pagamento_ult_30d,
            qtd_promessas_pagamento_ult_30d=qtd_promessas_pagamento_ult_30d,
        )

        probabilidade = df_resultado["probabilidade"].iloc[0]
        previsao = df_resultado["previsao"].iloc[0]
        mensagem = df_resultado["mensagem_final"].iloc[0]

        # ------------------------------------------------------------
        # CÁLCULO DAS DATAS E HORIZONTE
        # ------------------------------------------------------------
        agora = datetime.now()
        data_hora_atual = agora.strftime("%d-%m-%Y %H:%M:%S")

        # Lógica para determinar o próximo mês
        if agora.month == 12:
          ano_prox_mes = agora.year + 1
          prox_mes = 1
        else:
          ano_prox_mes = agora.year
          prox_mes = agora.month + 1

        _, ultimo_dia = calendar.monthrange(ano_prox_mes, prox_mes)

        inicio_horizonte = f"01-{prox_mes:02d}-{ano_prox_mes}"
        fim_horizonte = f"{ultimo_dia:02d}-{prox_mes:02d}-{ano_prox_mes}"

        # ------------------------------------------------------------
        # RENDERIZAÇÃO DOS RESULTADOS
        # ------------------------------------------------------------
        st.subheader("🎯 Resultado da Inferência")

        m_col1, m_col2 = st.columns(2)
        with m_col1:
          st.metric(
              label="Probabilidade do Evento", value=f"{probabilidade:.1%}"
          )

        with m_col2:
          if previsao == 1:
            st.error(f"**Classe Prevista:** {previsao}")
          else:
            st.success(f"**Classe Prevista:** {previsao}")

        # Informações temporais
        st.markdown(
            f"**Data/Hora do Processamento:** `{data_hora_atual}`  \n"
            f"**Horizonte da Previsão:** `{inicio_horizonte}` a"
            f" `{fim_horizonte}`"
        )

        # Mensagem do modelo
        if previsao == 1:
          st.warning(f"⚠️ **Detalhes:** {mensagem}")
        else:
          st.info(f"ℹ️ **Detalhes:** {mensagem}")

        # Nota de rodapé sobre os dados
        st.caption(
            "⚠️ *Nota: Os dados informados nesta tela são puramente hipotéticos"
            " e simulam informações coletadas no mês de referência atual para"
            " fins de teste do modelo.*"
        )

        # Inspecionar estrutura completa retornada
        with st.expander("🔍 Ver DataFrame retornado pela inferência"):
          st.dataframe(df_resultado)

      except Exception as e:
        st.error(f"❌ Erro ao realizar a inferência: {str(e)}")


# Caso execute diretamente
if __name__ == "__main__":
  render_modelo_preditivo_page()
