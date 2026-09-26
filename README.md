
---

O **Renova PJ** é uma solução analítica completa que utiliza Inteligência Artificial para otimizar a recuperação de crédito de Pessoas Jurídicas (PJ). A plataforma reúne um conjunto robusto de ferramentas integradas:

1. **Modelo Preditivo:** Avalia a probabilidade de um cliente regularizar seus débitos nos próximos 30 dias.
2. **Analytics Avançado:** Mapeia o perfil e o comportamento dos clientes, analisando variáveis como localização, setor, porte, tempo de relacionamento, uso do limite de cartão de crédito, saldo devedor e número de parcelas.
3. **Base de Conhecimento (RAG):** Centraliza e consulta dados estratégicos, como critérios para ofertas de renegociação, guias de atendimento e políticas de recuperação de crédito.
4. **Métricas de RAG:** Monitora a volumetria e a estrutura dos documentos indexados na base, exibindo contagem de tokens e palavras no escopo geral e por documento.
5. **Métricas do Assistente:** Acompanha indicadores de performance da IA baseados no *Golden Set*, mensurando fidelidade, relevância das respostas, precisão de contexto e taxa de revocação (*recall*).
6. **Assistente Autônomo:** O núcleo da solução. Integrado ao ecossistema da **IA Gemini**, ele consulta informações cadastrais e financeiras, recupera dados operacionais e sugere ações estratégicas em tempo real, respondendo a perguntas como:

* *Qual é o perfil do cliente?*
* *Qual é a probabilidade de regularização?*
* *Quais fatores influenciam essa previsão?*
* *Qual estratégia de renegociação é recomendada?*
* *Quais evidências e documentos sustentam a recomendação?*
* *Existem restrições políticas ou regulatórias para essa estratégia?*
Com essa abordagem de ponta a ponta — unindo preditividade, analytics, RAG e agentes autônomos —, o **Renova PJ** transforma a gestão de inadimplência em uma operação preditiva, ágil e altamente eficiente.
---

## 🔗 Links Úteis

* **Acesso Online (Streamlit Cloud):** [itaurecuperapj.streamlit.app](https://itaurecuperapj.streamlit.app/)
* **Repositório do Projeto:** [github.com/rogercsampaio/itau_recupera_pj](https://github.com/rogercsampaio/itau_recupera_pj)

---

## 📌 Funcionalidades Principais

* **Dashboard Analytics:** Visualização de indicadores globais da base e desempenho dos clientes incluindo informações financeiras, perfil e comportamentais.
* **Assistente Autonômo** Coração do aplicativo, funcionalidade capaz de oferecer recomendações, ações sobre a regulamentação dos clientes PJ com uma linguagem natural, usando um chat e Inteligência Artificial. Ela integra todas as demais funcionalidades. Foi usado o Google Gemini.
* **RAG/Documentos:** Funcionalidade de busca de documentos através de similariedade semântica, ou seja, usando palavras-chaves que não precisam ser exatamente as mesmas dos textos originais. Use para busca informações sobre: 1. critérios de ofertas de renegociação, conduta de atendimento, manual de canais e contato, politica de recuperação de crédito entre outros.
* **Modelo Preditivo:** Use para simular o resultado do modelo(probabilidade de regularização dentro dos próximos 30 dias) considerando as variavéis de entrada tais como: dias de atraso, quantidade de parcelas vencida entre outras
* **Métricas Assistente** Acompanhamento das métricas do assistente autônomo *faithfulness*, *answer_relevancy*, *context_precision* e *context_recall*. 

---

## 🛠️ Tecnologias e Pré-requisitos

* **Linguagem:** Python `3.10`
* **Gerenciador de Ambiente:** Anaconda / Miniconda (recomendado)
* **Framework Web:** Streamlit
* **Análise e Visualização:** Pandas, Matplotlib, Seaborn, Plotly
* **LLM / Avaliação:** Google Generative AI (`gemini-3.1-flash-lite`)

---
## 🛠️ Arquitetura
A arquitetura do Renova PJ foi projetada de forma modular, separando a interface gráfica (Streamlit), os serviços de suporte (RAG, Agente e ML), as bases de dados e os artefatos de modelo. Essa estrutura garante facilidade de manutenção e rastreabilidade dos componentes do projeto

```text
recupera_pj/
│
├── app.py                       # Ponto de entrada (módulo principal do Streamlit)
├── requirements.txt             # Dependências do projeto
│
├── abas/                        # Páginas da interface gráfica (Streamlit)
│   ├── 1_analytics.py           # Dashboard de indicadores gerais
│   ├── 2_rag_documentos.py     # Consulta e navegação na base de conhecimento
│   ├── 3_assistente_ia.py      # Interface do Assistente Autônomo
│   ├── 4_modelo_preditivo.py   # Visualização e análises do modelo ML
│   ├── 5_metricas_rag.py       # Dashboard de avaliação do RAG
│   └── 6_metricas_agente.py    # Dashboard de avaliação do Agente (LLM-as-a-Judge)
│
├── src/                         # Código backend e motores de execução
│   ├── rag/                     # Pipeline de vetorização e busca RAG
│   ├── assistente/              # Lógica e orquestração do Agente Autônomo
│   └── modelagem_preditiva/     # Scripts de pré-processamento e inferência
│
├── bases_tratadas/              # Datasets de entrada sanitizados
│   ├── clientes_limpos.csv     # Base de clientes tratada
│   ├── abt_regularizacao_30d.csv # Analytical Base Table (ABT) para o modelo
│   ├── documentos_textos.csv   # Textos extraídos e prontos para vetorização
│   └── golden_set.csv           # Conjunto de teste rotulado para avaliação do RAG
│
├── chroma_db/                   # Banco de dados vetorial persistido
├── modelos/                     # Objetos de ML treinados e serializados (.pkl/.json)
├── resultados/                  # Saídas das inferências e métricas salvas
│   ├── inferencias_clientes.csv # Previsões de propensão aplicadas aos clientes
│   └── metricas_agente.csv      # Resultados das avaliações do agente
│
├── code/                        # Scripts de suporte de infraestrutura
│   ├── config_log.py            # Configuração centralizada de logs
│   └── proxies.py               # Definições e rotas de proxy
│
├── logs/                        # Registros de log gerados durante a execução do app
├── notebooks/                   # Notebooks Jupyter de EDA, treino e avaliação
├── imagens/                     # Identidade visual e diagramas de arquitetura
└── uteis/                       # Utilitários auxiliares
```


### Arquitetura da Solução

A arquitetura do **Renova PJ** foi projetada de forma modular, separando a interface gráfica (Streamlit), os serviços de suporte (RAG, Agente e ML), as bases de dados e os artefatos de modelo. Essa estrutura garante facilidade de manutenção e rastreabilidade dos componentes do projeto.


### **Estrutura de Diretórios**

itau_recupera_pj/
│
├── app.py                      # Ponto de entrada (módulo principal do Streamlit)
├── requirements.txt            # Dependências do projeto
│
├── abas/                       # Páginas da interface gráfica (Streamlit)
│   ├── 1_analytics.py          # Dashboard de indicadores gerais
│   ├── 2_rag_documentos.py     # Consulta e navegação na base de conhecimento
│   ├── 3_assistente_ia.py      # Interface do Assistente Autônomo
│   ├── 4_modelo_preditivo.py   # Visualização e análises do modelo ML
│   ├── 5_metricas_rag.py       # Dashboard de avaliação do RAG
│   └── 6_metricas_agente.py    # Dashboard de avaliação do Agente (LLM-as-a-Judge)
│
├── src/ (ou componentes)       # Código backend e motores de execução
│   ├── rag/                    # Pipeline de vetorização e busca RAG
│   ├── assistente/             # Lógica e orquestração do Agente Autônomo
│   └── modelagem_preditiva/    # Scripts de pré-processamento e inferência
│
├── bases_tratadas/             # Datasets de entrada sanitizados
│   ├── clientes_limpos.csv     # Base de clientes tratada (sem valores nulos)
│   ├── abt_regularizacao_30d.csv # Analytical Base Table (ABT) para o modelo
│   ├── documentos_textos.csv   # Textos extraídos e prontos para vetorização
│   └── golden_set.csv          # Conjunto de teste rotulado para avaliação do RAG
│
├── chroma_db/                  # Banco de dados vetorial persistido
├── modelos/                    # Objetos de ML treinados e serializados (.pkl/.json)
├── resultados/                 # Saídas das inferências e métricas salvas
│   ├── inferencias_clientes.csv # Previsões de propensão aplicadas a todos os clientes
│   └── metricas_agente.csv      # Resultados das avaliações do agente
│
├── code/                       # Scripts de suporte de infraestrutura
│   ├── config_log.py           # Configuração centralizada de logs
│   └── proxies.py              # Definições e rotas de proxy
│
├── logs/                       # Registros de log gerados durante a execução do app
├── notebooks/                  # Notebooks Jupyter de EDA, treino e avaliação
├── imagens/                    # Identidade visual, artes do app e diagramas de arquitetura
└── uteis/                      # Utilitários auxiliares (extrator de texto, etc.)

---

### **Componentes Principais**

* **Interface Gráfica (`abas/`):** Estrutura modular no Streamlit onde cada arquivo `.py` representa uma aba funcional da aplicação, garantindo isolamento da camada de apresentação.
* **Motores de Execução:** Módulos dedicados ao processamento pesado do pipeline RAG, orquestração do agente autônomo de IA e execução das predições do XGBoost.
* **Persistência e Vetores (`chroma_db/` & `bases_tratadas/`):** Utilização do ChromaDB para armazenamento e busca de *embeddings*, aliado a um repositório centralizado de dados sanitizados para treino e inferência.
* **Governabilidade e MLOps (`code/`, `logs/` & `resultados/`):** Rastreabilidade garantida através de logs de execução, controle de versão de modelos treinados e armazenamento persistente das métricas de avaliação.

## 🚀 Como Rodar o Projeto Localmente
Não foi usado ferramenta de mock para criação. Todas as ferramentas podem ser testados conversando com a Inteligência Artificial pro meio de tools.
Importante ressaltar que para rodar localmente obrigatoriamente deve ter o python 3.10 e ambiente anaconda. Siga os passos abaixo para clonar o repositório, configurar o ambiente virtual no Anaconda e executar a aplicação localmente.

### 1. Clonar o Repositório
```bash
git clone [https://github.com/rogercsampaio/itau_recupera_pj.git](https://github.com/rogercsampaio/itau_recupera_pj.git)
cd itau_recupera_pj
```

###  2. Cria o ambiente virtual dedicado
```bash
conda create --name renova_pj python=3.10 -y
```
### 3. Ativa o ambiente
```bash
conda activate renova_pj
```
###  4. Instale as dependências
Com o ambiente renova_pj ativado, instale todos os pacotes necessários a partir do arquivo na raiz:
```bash
pip install -r requirements.txt
```
### 5. Configurar a Chave de API da Google
Importante: para ser executar o projeto localmente deve ser criado um tóken do Google Gemini. Acesse o link e veja como: https://aistudio.google.com/prompts/new_chat
Se for executar a rotina de avaliação com a API do Gemini localmente, defina a variável de ambiente:
```bash
No terminal do conda antes de executar a app rode.
conda env config vars set GEMINI_API_KEY="SUA_CHAVE_AQUI"
```
Depois desative seu ambiente e ative novamente. Para desativar use:
```bash
conda deactivate

```
Para ativar use:
```bash
conda activate <nome_ambiente>
```

### 6. Executar a Aplicação Streamlit
```
Certifique-se de estar no diretório raiz do projeto e execute:
streamlit run app.py
```
O Streamlit abrirá automaticamente no seu navegador padrão no endereço http://localhost:8501.
⚠️ Possíveis Problemas e Como Resolver


## 🚀 Como Rodar o Projeto Online
Não requer instalação ou configuração prévias, bastando somente acessar o link abaixo. O aplicativo pode ser acessado por computadores ou celulares via navegador. Não querer configuração de tóken também.
https://itaurecuperapj.streamlit.app/

## 1. Erro de Limite de Cota da API (HTTP 429 - Rate Limit)
Sintoma: O console exibe o erro 429 You exceeded your current quota.

Causa: A chave gratuita (Free Tier) da API do Gemini possui um limite estrito de 15 requisições por minuto (RPM).

Solução: O código já possui tratamento com pausa preventiva (time.sleep(4.5)) e retry automático. Se o erro ainda ocorrer, aguarde cerca de 1 minuto para a API renovar a janela temporal de requisições.
Caso esteja executando online após exceder esse limite, espere mais de um minuto para tentar realizar a próxima consulta.  Observação: Uma possível melhoria do aplicativo é TRATAR esse comportamento via código alertando para o usuário.

## Solução Analítica

### Estratédias do modelo preditivo
A modelagem de propensão à regularização foi estruturada a partir de um split temporal rigoroso, garantindo a simulação real de uso em produção e evitando vazamento de dados (data leakage)

| # | Feature | Categoria | Descrição / Significado Operacional |
| --- | --- | --- | --- |
| 1 | `qtd_interacoes_ult_30d` | Comportamento / Engajamento | Volume total de interações nos últimos 30 dias. |
| 2 | `dias_desde_ultima_interacao` | Comportamento / Recência | Tempo (em dias) decorrido desde o último contato registrado. |
| 3 | `qtd_interacoes_ativas_ult_30d` | Comportamento / Ativa | Frequência de contatos iniciados pela instituição financeira. |
| 4 | `qtd_interacoes_receptivas_ult_30d` | Comportamento / Receptiva | Frequência de contatos espontâneos iniciados pelo cliente. |
| 5 | `qtd_canais_distintos_ult_30d` | Comportamento / Canais | Diversidade de canais de atendimento utilizados no período. |
| 6 | `teve_promessa_pagamento_ult_30d` | Negociação / Acordo | Indicador binário de registro de promessa de pagamento. |
| 7 | `qtd_promessas_pagamento_ult_30d` | Negociação / Acordo | Volume total de promessas de pagamento formalizadas. |
| 8 | `utilizacao_limite_pct` | Perfil Financeiro / Crédito | Percentual de comprometimento do limite de crédito disponível. |
| 9 | `qtd_contratos_ativos` | Perfil Financeiro / Carteira | Total de produtos/contratos de crédito ativos na instituição. |
| 10 | `qtd_parcelas_vencidas` | Inadimplência / Severidade | Quantidade de parcelas em aberto/atrasadas. |
| 11 | `dias_atraso` | Inadimplência / Severidade | Dias corridos de atraso do contrato de maior risco. |

| Métrica | Treino (In-Sample) | Teste (Out-of-Time) | Delta (Drop) |
| --- | --- | --- | --- |
| **Acurácia** | 0.70 | 0.62 | -0.08 |
| **Precisão** | 0.69 | 0.54 | -0.15 |
| **Recall (Sensibilidade)** | 0.78 | 0.72 | -0.06 |
| **F1-Score** | 0.73 | 0.61 | -0.12 |



Destaque: O modelo prioriza o Recall (72% no teste Out-of-Time), maximizando a captura de clientes com alto potencial de regularização para acionamento das equipes de cobrança. Esta versão representa a Baseline (v1.0) da solução. Para otimização contínua do modelo, foram identificadas as seguintes alavancas de melhoria: Enriquecimento de Features (Feature Engineering), Mitigação de Overfitting e Calibração, Exploração de Novos Algoritmos


### Limitações/Retrições
1. Métricas Assistente. O atingimento de **100%** nas métricas do agente (*faithfulness*, *answer_relevancy*, *context_precision* e *context_recall*) reflete a alta assertividade do RAG sobre uma base reduzida a apenas **5 documentos**, onde o resgate vetorial captura o contexto exato sem alucinações. Contudo, esse resultado perfeito aciona um **sinal de alarme**: embora possa indicar pleno funcionamento, também pode mascarar *overfitting* ou falta de sensibilidade dos testes, sendo um ponto crítico que cabe investigação em cenários mais complexos. Essa abordagem enxuta foi uma decisão estratégica para a **Versão 1.0 (MVP)**, desenvolvida sob uma **janela de tempo extremamente reduzida**, com o objetivo principal de entregar uma arquitetura *end-to-end* funcional e validada dentro do prazo.

2. Modelagem preditiva. O notebook do projeto contém todo o pipeline de modelagem preditiva — desde a preparação dos dados até a geração das previsões finalizadas —, estruturado de forma sequencial em células separadas. Em um cenário ideal de produção, esse fluxo seria totalmente modularizado em scripts e pipelines reutilizáveis (`.py`); no entanto, diante do **prazo restrito de desenvolvimento**, a prioridade estratégica foi garantir uma versão *end-to-end* totalmente funcional e integrada ao aplicativo Streamlit, deixando o *refactoring* de código como uma evolução natural para a próxima versão.

O monitoramento de *data e concept drift* não foi implementado nesta etapa devido ao **prazo restrito de desenvolvimento**, que exigiu priorizar a entrega de uma primeira versão funcional da modelagem preditiva e da aplicação web. Trata-se de uma evolução natural e essencial para as próximas versões, onde a esteira de MLOps será integrada para acompanhar o comportamento dos dados em produção, detectar degradações na performance do modelo ao longo do tempo e disparar gatilhos automáticos de retreinamento.

Durante a Análise Exploratória de Dados (EDA), identificaram-se as variáveis `score_regularizacao_legada` (cuja origem e governança precisam ser auditadas) e `score_cobranca_atualizado`, que apresenta forte suspeita de ***data leakage*** por potencialmente incorporar dados dentro da janela do *target*; por isso, como medida de segurança técnica nesta Versão 1.0, ambas foram mantidas fora da modelagem, ficando a investigação aprofundada de sua temporalidade e utilidade preditiva mapeada para os próximos passos.

No notebook do projeto podem ser observados alguns rascunhos e análises exploratórias iniciais, resultantes da reorganização interna do desenvolvimento. Ainda assim, foi mantida uma estrutura padronizada e bem comentada para organizar os fluxos, divididos em **Objetivo**, **Importações**, **Definição de Funções** e **Rotina Principal**. Diante do **prazo restrito**, essa abordagem garantiu a rastreabilidade do código e a entrega ágil de uma solução funcional integrada ao aplicativo.

