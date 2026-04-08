# Luna

Aplicação em **Streamlit** para análise de tempos operacionais de entrega, com foco em:

- qualidade da base
- expurgo de registros inconsistentes
- mediana de atendimento por cliente
- cálculo de **janelas de atendimento**
- exportação consolidada para Excel

Esta versão foi estruturada para uma leitura mais executiva, com melhorias de **UI/UX** e organização mais clara das etapas de processamento.

---

## Objetivo

O Luna foi criado para transformar bases operacionais de entrega em uma análise prática para tomada de decisão.

A aplicação permite:

- padronizar a base de entrada
- validar colunas obrigatórias
- identificar inconsistências
- aplicar regras operacionais
- calcular medianas por cliente
- estimar janelas de atendimento por cliente
- exportar os resultados em múltiplas abas

---

## Principais recursos

### 1. Validação e padronização da base
Antes de processar os dados, o sistema:

- padroniza nomes de colunas
- verifica se o schema atende ao mínimo necessário
- identifica colunas ausentes
- gera visão de inconsistências

### 2. Regras operacionais
Após a validação, o Luna aplica filtros e regras de negócio, como:

- expurgo por tempo mínimo
- separação de anomalias por tempo máximo
- tratamento de base válida para análises seguintes

### 3. Medianas por cliente
Para cada cliente, o sistema calcula:

- quantidade de apontamentos
- mediana de tempo de atendimento
- método aplicado no cálculo

A lógica considera:

- uso dos últimos `N` eventos
- tratamento para clientes com poucos apontamentos
- ajuste percentual opcional

### 4. Janelas de atendimento
A aplicação também calcula a janela operacional de atendimento por cliente, considerando:

- percentual inicial da janela
- percentual final da janela
- horário estimado de início
- horário estimado de fim
- duração da janela
- cobertura amostral
- período predominante

### 5. Exportação
Os resultados podem ser exportados em Excel com múltiplas abas:

- Base Bruta
- Base Validada
- Inconsistencias
- Expurgados
- Anomalias
- Medianas Cliente
- Janelas Atendimento

---

## Estrutura do projeto

```text
luna/
├── app.py
├── config.py
├── data/
├── exports/
├── assets/
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── data_transformer.py
    ├── engine.py
    └── schema.py