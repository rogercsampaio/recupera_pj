# Contém uma série de métodos úteis de modelagem preditiva como, por exemplo:
# 1 - Carregamento do modelo treinado
# 2 - Inferir avulso dado um determinado cliente entre outros.
# São métodos acionados pela interface aba modelagem

import os
from pathlib import Path
import joblib
from loguru import logger
import pandas as pd


def obter_raiz_projeto(diretorio_atual: Path = None) -> Path:
  """Localiza a raiz do projeto buscando por arquivos marcadores padrão (.git, requirements.txt, pyproject.toml)

  ou navegando pelos diretórios pai.
  """
  if diretorio_atual is None:
    diretorio_atual = Path(__file__).resolve().parent

  # Procura por arquivos comuns da raiz do projeto
  for marcador in [
      '.git',
      'requirements.txt',
      'pyproject.toml',
      'setup.py',
      'README.md',
  ]:
    for parent in [diretorio_atual] + list(diretorio_atual.parents):
      if (parent / marcador).exists():
        return parent

  # Caso não encontre nenhum marcador, assume o diretório pai atual
  return diretorio_atual


# Define a raiz do projeto de forma dinâmica e absoluta
RAIZ_PROJETO = obter_raiz_projeto()


def carregar_modelo(nome_arquivo: str, pasta: str = 'modelos'):
  """Carrega um modelo previamente salvo localizado na pasta 'modelos' na raiz do projeto.

  Parâmetros
  ----------
  nome_arquivo : str
      Nome do arquivo salvo (ex: 'xgboost_tunado_regularizacao.pkl').
  pasta : str
      Pasta onde o modelo está dentro da raiz (padrão: 'modelos').

  Retorno
  -------
  modelo
      Objeto do modelo carregado.
  """
  # Garantia adicional: Constrói o caminho absoluto sempre apontando para <RAIZ>/modelos/<arquivo>
  caminho = RAIZ_PROJETO / pasta / nome_arquivo

  try:
    if not caminho.exists():
      raise FileNotFoundError(
          f'Modelo não encontrado no caminho absoluto: {caminho}'
      )

    modelo = joblib.load(caminho)
    logger.info(f'Modelo carregado com sucesso de: {caminho}')
    return modelo

  except FileNotFoundError as e:
    logger.error(f'Erro de arquivo: {e}')
    raise

  except Exception as e:
    logger.error(
        f'Erro inesperado ao carregar o modelo em {caminho}: {str(e)}'
    )
    raise


def validar_entradas(dados: dict):
  """Valida as regras de negócio para as variáveis de entrada antes de enviar ao modelo."""
  # 1. Validação de teve_promessa_pagamento_ult_30d (apenas 0.0, 1.0, 0 ou 1)
  promessa = dados['teve_promessa_pagamento_ult_30d']
  if promessa not in (0.0, 1.0, 0, 1):
    raise ValueError(
        f"'teve_promessa_pagamento_ult_30d' deve ser 0 ou 1. Recebido:"
        f' {promessa}'
    )

  # 2. Validação de utilizacao_limite_pct (não pode ser negativo e limite razoável)
  limite = dados['utilizacao_limite_pct']
  if limite < 0:
    raise ValueError(
        f"'utilizacao_limite_pct' não pode ser negativo. Recebido: {limite}"
    )
  if limite > 2.0:
    raise ValueError(
        f"'utilizacao_limite_pct' acima do limite razoável (max: 2.0). Recebido:"
        f' {limite}'
    )

  # 3. Validação de dias_desde_ultima_interacao (pode ser >= 0 ou -1 para 'sem interação')
  dias_interacao = dados['dias_desde_ultima_interacao']
  if dias_interacao < -1:
    raise ValueError(
        f"'dias_desde_ultima_interacao' deve ser >= 0 ou -1 (para 'sem"
        f" interação'). Recebido: {dias_interacao}"
    )

  # 4. Validação para todas as outras variáveis numéricas de quantidade/dias
  outras_features_positivas = [
      'dias_atraso',
      'qtd_parcelas_vencidas',
      'qtd_contratos_ativos',
      'qtd_interacoes_ult_30d',
      'qtd_canais_distintos_ult_30d',
      'qtd_interacoes_ativas_ult_30d',
      'qtd_interacoes_receptivas_ult_30d',
      'qtd_promessas_pagamento_ult_30d',
  ]

  for feature in outras_features_positivas:
    valor = dados[feature]
    if valor < 0:
      raise ValueError(
          f"A feature '{feature}' não pode ser negativa. Recebido: {valor}"
      )


def inferir_cliente_avulso(
    dias_atraso: int,
    qtd_parcelas_vencidas: int,
    utilizacao_limite_pct: float,
    qtd_contratos_ativos: int,
    qtd_interacoes_ult_30d: int,
    dias_desde_ultima_interacao: int,
    qtd_canais_distintos_ult_30d: int,
    qtd_interacoes_ativas_ult_30d: int,
    qtd_interacoes_receptivas_ult_30d: int,
    teve_promessa_pagamento_ult_30d: float,
    qtd_promessas_pagamento_ult_30d: int,
    nome_modelo: str = 'xgboost_tunado_regularizacao.pkl',
) -> pd.DataFrame:
  """Realiza a validação, inferência e gera o relatório do cliente."""

  dados_cliente = {
      'dias_atraso': dias_atraso,
      'qtd_parcelas_vencidas': qtd_parcelas_vencidas,
      'utilizacao_limite_pct': utilizacao_limite_pct,
      'qtd_contratos_ativos': qtd_contratos_ativos,
      'qtd_interacoes_ult_30d': qtd_interacoes_ult_30d,
      'dias_desde_ultima_interacao': dias_desde_ultima_interacao,
      'qtd_canais_distintos_ult_30d': qtd_canais_distintos_ult_30d,
      'qtd_interacoes_ativas_ult_30d': qtd_interacoes_ativas_ult_30d,
      'qtd_interacoes_receptivas_ult_30d': qtd_interacoes_receptivas_ult_30d,
      'teve_promessa_pagamento_ult_30d': teve_promessa_pagamento_ult_30d,
      'qtd_promessas_pagamento_ult_30d': qtd_promessas_pagamento_ult_30d,
  }

  try:
    # Passo 1: Executa as regras de validação
    validar_entradas(dados_cliente)

    # Passo 2: Prepara os dados para a predição
    df_cliente = pd.DataFrame([dados_cliente])

    # Passo 3: Carrega o modelo localizado na pasta 'modelos' na raiz do projeto
    modelo_inferencia = carregar_modelo(nome_modelo)

    # Passo 4: Gera previsões
    probabilidade = float(modelo_inferencia.predict_proba(df_cliente)[:, 1][0])
    previsao = int(modelo_inferencia.predict(df_cliente)[0])

    # Passo 5: Mensagem descritiva
    if previsao == 1:
      mensagem = (
          f'Cliente classificado na Classe 1 (Probabilidade:'
          f' {probabilidade:.1%})'
      )
    else:
      mensagem = (
          f'Cliente classificado na Classe 0 (Probabilidade:'
          f' {probabilidade:.1%})'
      )

    # Passo 6: Retorno estruturado
    df_resultado = df_cliente.copy()
    df_resultado['probabilidade'] = probabilidade
    df_resultado['previsao'] = previsao
    df_resultado['mensagem_final'] = mensagem

    return df_resultado

  except ValueError as err_val:
    logger.error(f'Falha na validação dos dados de entrada: {err_val}')
    raise

  except Exception as err:
    logger.error(f'Erro durante o processo de inferência: {err}')
    raise