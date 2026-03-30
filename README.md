# Luna Deploy — v2

Aplicação **Streamlit** para análise de tempos operacionais por cliente, com:

- validação de schema de entrada;
- padronização de colunas (aliases);
- classificação de inconsistências;
- regras de expurgo/anomalia por tempo;
- cálculo de medianas por cliente;
- painel visual por cliente;
- exportação de resultados em Excel e ZIP (CSVs).

---

## O que o projeto executa (v2)

Na versão **v2**, o fluxo principal é:

1. Selecionar unidade e parâmetros operacionais na sidebar.
2. Carregar base da unidade via Google Drive (com `st.secrets`).
3. Padronizar colunas e validar estrutura mínima obrigatória.
4. Transformar dados e separar:
   - válidos;
   - inconsistências;
   - expurgados;
   - anomalias.
5. Calcular medianas por cliente e KPIs globais.
6. Exibir resultados em abas e permitir exportação.

Abas disponíveis na interface:

- **Base**
- **Validação**
- **Processamento**
- **Painel do Cliente**
- **Resultados**
- **Exportação**

---

## Estrutura resumida

- `app.py`: aplicação principal (UI + orquestração do pipeline).
- `src/data_loader.py`: carregamento do CSV via Google Drive.
- `src/schema.py`: schema oficial, aliases, validação e sugestões.
- `src/data_transformer.py`: transformação e classificação operacional.
- `src/engine.py`: medianas, KPIs e exportações.
- `config.py`: constantes e diretórios base.

---

## Requisitos

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## Configuração de secrets

Crie `.streamlit/secrets.toml` com:

```toml
[gcp_service_account]
# credenciais da service account do Google Cloud
# (campos padrão do JSON da conta de serviço)

[drive_files]
# mapa unidade -> file_id no Google Drive
MGA = "FILE_ID_MGA"
GPV = "FILE_ID_GPV"
PG  = "FILE_ID_PG"
NP  = "FILE_ID_NP"
```

> Sem esses blocos, o carregamento dos arquivos por unidade não funciona.

---

## Como executar localmente

```bash
streamlit run app.py
```

---

## Versão

Versão atual documentada neste repositório: **v2**.
