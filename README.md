# Case Recuperação de Crédito PJ — ML e IA Generativa

Pacote de avaliação técnica para seleção de Cientista de Dados Sênior. O material combina modelagem supervisionada, recomendação explicável, IA Generativa, RAG, Tool Calling e arquitetura produtiva.

## Estrutura

```text
case_recuperacao_credito_pj/
├── ENUNCIADO_CASE.md
├── DICIONARIO_DADOS.md
├── README.md
├── clientes.csv
├── interacoes.csv
├── generate_synthetic_data.py
├── mock_tools.py
├── agent_skeleton.py
├── requirements.txt
├── documentos_rag/
│   ├── politica_recuperacao_pj.txt
│   ├── faq_negociacao_pj.txt
│   ├── manual_canais_contato.txt
│   ├── criterios_ofertas_renegociacao.txt
│   └── guia_conduta_atendimento.txt
├── tests/
│   └── test_smoke.py
└── _interno/
    └── GUIA_AVALIACAO.md
```

O diretório `_interno/` é destinado à banca e deve ser removido antes do envio ao candidato, caso se deseje ocultar a rubrica.

## Dados

- `clientes.csv`: 800 clientes PJ sintéticos e target `regularizou_30d`.
- `interacoes.csv`: histórico textual sintético, anterior à data de referência de cada cliente.
- `documentos_rag/`: cinco documentos fictícios usados como base de conhecimento.

Os dados são reproduzíveis com seed fixa. Para regenerá-los:

```bash
python generate_synthetic_data.py
```

## Execução rápida

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python agent_skeleton.py --cliente PJ0001 --pergunta "Qual estratégia é recomendada e quais documentos a sustentam?"
```

Teste das ferramentas:

```bash
python -m unittest discover -s tests -v
```

## Uso das ferramentas

```python
from mock_tools import (
    consultar_cliente,
    consultar_score_regularizacao,
    buscar_historico_interacoes,
    buscar_documentacao,
    recomendar_acao_basica,
)

cliente = consultar_cliente("PJ0001")
score = consultar_score_regularizacao("PJ0001")
historico = buscar_historico_interacoes("PJ0001", limite=5)
documentos = buscar_documentacao("regras para parcelamento e desconto", top_k=3)
acao = recomendar_acao_basica("PJ0001")
```

As funções retornam objetos serializáveis em JSON e funcionam localmente. A busca documental utiliza um baseline lexical intencionalmente simples, adequado para ser substituído por embeddings, busca híbrida e reranking.

## Aviso

Todos os nomes, dados, políticas, documentos, regras, scores e resultados são fictícios. Nenhum dado real foi utilizado.
