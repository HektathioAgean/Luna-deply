import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import APP_TITLE, AVAILABLE_UNITS, LAYOUT
from src.data_loader import load_unit_file, list_available_unit_files
from src.data_transformer import aplicar_regras_operacionais, transform_base
from src.engine import (
    build_kpis,
    calcular_medianas_por_cliente,
    exportar_excel,
    format_seconds,
    format_seconds_hhmm,
    montar_evolucao_cliente,
)
from src.schema import (
    analyze_schema,
    get_aliases_dataframe,
    get_schema_dataframe,
    schema_report_to_dict,
    standardize_columns,
    suggest_missing_columns,
)

os.environ["OMP_NUM_THREADS"] = "1"

st.set_page_config(
    page_title=APP_TITLE,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def executar_pipeline(
    unidade: str,
    tempo_min_expurgo: int,
    tempo_max_anomalia: int,
    eventos_previos: int,
    minimo_apontamentos: int,
    tempo_padrao_poucos_apontamentos: int,
    ajuste_percentual: int,
) -> dict:
    base_bruta = load_unit_file(unidade)

    if base_bruta is None or base_bruta.empty:
        return {
            "base_bruta": pd.DataFrame(),
            "base_padronizada": pd.DataFrame(),
            "schema_report": None,
            "relatorio_validacao": None,
            "inconsistencias": pd.DataFrame(),
            "processados": pd.DataFrame(),
            "expurgados": pd.DataFrame(),
            "anomalias": pd.DataFrame(),
            "medianas": pd.DataFrame(),
            "kpis": {},
            "erro": "A base está vazia ou não pôde ser carregada.",
        }

    base_padronizada = standardize_columns(base_bruta)
    schema_report = analyze_schema(base_padronizada)
    relatorio_validacao = schema_report_to_dict(schema_report)

    base_validos, inconsistencias = transform_base(base_padronizada)

    processados, expurgados, anomalias = aplicar_regras_operacionais(
        base_validos,
        tempo_min_expurgo=tempo_min_expurgo,
        tempo_max_anomalia=tempo_max_anomalia,
    )

    medianas = calcular_medianas_por_cliente(
        df=processados,
        eventos_previos=eventos_previos,
        minimo_apontamentos=minimo_apontamentos,
        tempo_padrao_poucos_apontamentos=tempo_padrao_poucos_apontamentos,
        ajuste_percentual=ajuste_percentual,
    )

    kpis = build_kpis(
        base_bruta=base_padronizada,
        base_validos=processados,
        inconsistencias=inconsistencias,
        expurgados=expurgados,
        anomalias=anomalias,
        medianas=medianas,
    )

    return {
        "base_bruta": base_bruta,
        "base_padronizada": base_padronizada,
        "schema_report": schema_report,
        "relatorio_validacao": relatorio_validacao,
        "inconsistencias": inconsistencias,
        "processados": processados,
        "expurgados": expurgados,
        "anomalias": anomalias,
        "medianas": medianas,
        "kpis": kpis,
        "erro": None,
    }


def gerar_figura_evolucao(df_evolucao: pd.DataFrame, mediana_cliente: float, cliente: str):
    fig, ax = plt.subplots(figsize=(14, 6))

    entregas = df_evolucao[df_evolucao["Tipo"] == "Entrega"].copy()
    entregas["Data_Label"] = pd.to_datetime(entregas["Data_Chegada"], errors="coerce").dt.strftime("%d/%m/%Y")

    x_linha = list(range(len(entregas)))
    x_mediana = len(entregas)

    ax.plot(
        x_linha,
        entregas["Tempo_Sec"].tolist(),
        marker="o",
        linewidth=2,
    )

    ax.bar(
        [x_mediana],
        [mediana_cliente],
        width=0.6,
    )

    for i, y in enumerate(entregas["Tempo_Sec"].tolist()):
        ax.annotate(
            format_seconds_hhmm(y),
            (i, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )

    ax.annotate(
        format_seconds_hhmm(mediana_cliente),
        (x_mediana, mediana_cliente),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

    x_labels = entregas["Data_Label"].tolist() + ["Mediana"]
    ax.set_xticks(list(range(len(x_labels))))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_title(f"Evolução dos tempos | Cliente {cliente}")
    ax.set_xlabel("Datas das entregas")
    ax.set_ylabel("Tempo de entrega (segundos)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    return fig


def _parametros_atuais(
    unidade: str,
    tempo_min_expurgo: int,
    tempo_max_anomalia: int,
    eventos_previos: int,
    minimo_apontamentos: int,
    tempo_padrao_poucos_apontamentos: int,
    ajuste_percentual: int,
) -> dict:
    return {
        "unidade": unidade,
        "tempo_min_expurgo": int(tempo_min_expurgo),
        "tempo_max_anomalia": int(tempo_max_anomalia),
        "eventos_previos": int(eventos_previos),
        "minimo_apontamentos": int(minimo_apontamentos),
        "tempo_padrao_poucos_apontamentos": int(tempo_padrao_poucos_apontamentos),
        "ajuste_percentual": int(ajuste_percentual),
    }


def main() -> None:
    st.title("Luna")
    st.caption("Análise de tempos operacionais")

    if "luna_processar_solicitado" not in st.session_state:
        st.session_state["luna_processar_solicitado"] = False
    if "luna_parametros_processados" not in st.session_state:
        st.session_state["luna_parametros_processados"] = None

    available_files = list_available_unit_files()
    unit_options = available_files if available_files else AVAILABLE_UNITS

    with st.sidebar:
        st.header("Entrada")

        unidade = st.selectbox(
            "Selecione a unidade",
            options=unit_options,
            index=0,
        )

        st.caption("Unidades configuradas para leitura no Google Drive:")
        if available_files:
            st.write(available_files)
        else:
            st.warning(
                "Nenhuma unidade foi encontrada em `[drive_files]` no "
                "arquivo `.streamlit/secrets.toml`."
            )

        st.header("Parâmetros")

        tempo_min_expurgo = st.number_input(
            "Tempo mínimo de expurgo (segundos)",
            min_value=1,
            value=300,
            step=1,
        )

        tempo_max_anomalia = st.number_input(
            "Tempo máximo para anomalia (segundos)",
            min_value=1,
            value=21600,
            step=1,
        )

        eventos_previos = st.number_input(
            "Eventos prévios",
            min_value=1,
            value=10,
            step=1,
        )

        minimo_apontamentos = st.number_input(
            "Mínimo de apontamentos por cliente",
            min_value=1,
            value=4,
            step=1,
        )

        tempo_padrao_poucos_apontamentos = st.number_input(
            "Tempo padrão para poucos apontamentos (segundos)",
            min_value=1,
            value=600,
            step=1,
        )

        ajuste_percentual = st.slider(
            "Ajuste percentual",
            min_value=-20,
            max_value=100,
            value=0,
            step=1,
        )

        parametros_sidebar = _parametros_atuais(
            unidade=unidade,
            tempo_min_expurgo=tempo_min_expurgo,
            tempo_max_anomalia=tempo_max_anomalia,
            eventos_previos=eventos_previos,
            minimo_apontamentos=minimo_apontamentos,
            tempo_padrao_poucos_apontamentos=tempo_padrao_poucos_apontamentos,
            ajuste_percentual=ajuste_percentual,
        )

        processar = st.button("Processar base", use_container_width=True)

        if processar:
            st.session_state["luna_processar_solicitado"] = True
            st.session_state["luna_parametros_processados"] = parametros_sidebar.copy()

    tab_base, tab_validacao, tab_processamento, tab_resultados, tab_evolucao, tab_exportacao = st.tabs(
        ["Base", "Validação", "Processamento", "Resultados", "Evolução Cliente", "Exportação"]
    )

    if not st.session_state.get("luna_processar_solicitado", False):
        with tab_base:
            st.info("Selecione a unidade, ajuste os parâmetros e clique em 'Processar base'.")
        return

    parametros_processados = st.session_state.get("luna_parametros_processados")
    if not parametros_processados:
        with tab_base:
            st.info("Selecione a unidade, ajuste os parâmetros e clique em 'Processar base'.")
        return

    if parametros_sidebar != parametros_processados:
        st.warning(
            "Os resultados exibidos correspondem ao último processamento executado. "
            "Se quiser aplicar os parâmetros atuais, clique novamente em 'Processar base'."
        )

    try:
        resultado = executar_pipeline(**parametros_processados)
    except Exception as exc:
        st.error(str(exc))
        return

    if resultado.get("erro"):
        st.error(resultado["erro"])
        return

    base_bruta = resultado["base_bruta"]
    base_padronizada = resultado["base_padronizada"]
    schema_report = resultado["schema_report"]
    relatorio_validacao = resultado["relatorio_validacao"]
    inconsistencias = resultado["inconsistencias"]
    processados = resultado["processados"]
    expurgados = resultado["expurgados"]
    anomalias = resultado["anomalias"]
    medianas = resultado["medianas"]
    kpis = resultado["kpis"]

    with tab_base:
        st.subheader("Visão da base")

        col1, col2, col3 = st.columns(3)
        col1.metric("Linhas brutas", len(base_bruta))
        col2.metric("Colunas encontradas", relatorio_validacao["total_columns"])
        col3.metric("Estrutura válida", "Sim" if relatorio_validacao["is_valid"] else "Não")

        st.dataframe(base_padronizada.head(100), use_container_width=True, height=450)

    with tab_validacao:
        st.subheader("Validação estrutural")

        if relatorio_validacao["is_valid"]:
            st.success("Estrutura válida para processamento.")
        else:
            st.error("Estrutura inválida.")
            st.write("Colunas obrigatórias ausentes:")
            st.write(relatorio_validacao["required_missing"])

            sugestoes = suggest_missing_columns(schema_report)
            if sugestoes:
                st.markdown("### Sugestões")
                for sugestao in sugestoes:
                    st.write(f"- {sugestao}")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Colunas obrigatórias encontradas")
            st.dataframe(
                {"Colunas": relatorio_validacao["required_found"]},
                use_container_width=True,
                height=180,
            )

        with col2:
            st.markdown("### Colunas extras não reconhecidas")
            st.dataframe(
                {"Colunas": relatorio_validacao["unknown_columns"]},
                use_container_width=True,
                height=180,
            )

        if relatorio_validacao["duplicate_standardized_columns"]:
            st.warning("Foram detectadas colunas duplicadas após a padronização:")
            st.write(relatorio_validacao["duplicate_standardized_columns"])

        st.markdown("### Mapeamento aplicado")
        st.dataframe(
            relatorio_validacao["mapping_preview"],
            use_container_width=True,
            height=320,
        )

        st.markdown("### Schema oficial")
        st.dataframe(
            get_schema_dataframe(),
            use_container_width=True,
            height=260,
        )

        st.markdown("### Aliases reconhecidos")
        st.dataframe(
            get_aliases_dataframe(),
            use_container_width=True,
            height=320,
        )

    with tab_processamento:
        st.subheader("KPIs do processamento")

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.metric("Linhas válidas", kpis["linhas_validas"])
        col2.metric("Inconsistências", kpis["inconsistencias"])
        col3.metric("Expurgados", kpis["expurgados"])
        col4.metric("Anomalias", kpis["anomalias"])
        col5.metric("Clientes únicos", kpis["clientes_unicos"])
        col6.metric("Mediana global", kpis["mediana_global_fmt"])

        st.subheader("Resumo")
        st.write(
            {
                "unidade": parametros_processados["unidade"],
                "linhas_brutas": kpis["linhas_brutas"],
                "linhas_validas": kpis["linhas_validas"],
                "inconsistencias": kpis["inconsistencias"],
                "expurgados": kpis["expurgados"],
                "anomalias": kpis["anomalias"],
                "clientes_unicos": kpis["clientes_unicos"],
                "mediana_global": kpis["mediana_global_fmt"],
            }
        )

    with tab_resultados:
        st.subheader("Medianas por cliente")
        st.dataframe(medianas, use_container_width=True, height=320)

        st.subheader("Inconsistências")
        st.dataframe(inconsistencias, use_container_width=True, height=240)

        st.subheader("Expurgados")
        st.dataframe(expurgados, use_container_width=True, height=240)

        st.subheader("Anomalias")
        st.dataframe(anomalias, use_container_width=True, height=240)

    evolucao_cliente = pd.DataFrame()

    with tab_evolucao:
        st.subheader("Evolução por cliente")

        if processados.empty or medianas.empty:
            st.info("Não há dados processados para exibir a evolução por cliente.")
        else:
            clientes = (
                processados["Cod_Cliente"]
                .dropna()
                .astype(str)
                .str.strip()
                .sort_values()
                .unique()
                .tolist()
            )

            if not clientes:
                st.info("Nenhum cliente disponível para análise.")
            else:
                cliente_default = clientes[0]
                if "cliente_evolucao" not in st.session_state or st.session_state["cliente_evolucao"] not in clientes:
                    st.session_state["cliente_evolucao"] = cliente_default

                cliente = st.selectbox(
                    "Selecione o cliente",
                    options=clientes,
                    key="cliente_evolucao",
                )

                evolucao_cliente, mediana_cliente = montar_evolucao_cliente(
                    base_validos=processados,
                    medianas=medianas,
                    cliente=cliente,
                )

                if evolucao_cliente.empty:
                    st.warning("Não foi possível montar a evolução para o cliente selecionado.")
                else:
                    col1, col2 = st.columns(2)
                    col1.metric("Cliente", cliente)
                    col2.metric("Mediana do cliente", format_seconds(mediana_cliente))

                    fig = gerar_figura_evolucao(evolucao_cliente, mediana_cliente, cliente)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    tabela_exibicao = evolucao_cliente.copy()
                    if "Data_Chegada" in tabela_exibicao.columns:
                        tabela_exibicao["Data_Chegada"] = pd.to_datetime(
                            tabela_exibicao["Data_Chegada"], errors="coerce"
                        ).dt.strftime("%d/%m/%Y")
                        tabela_exibicao["Data_Chegada"] = tabela_exibicao["Data_Chegada"].fillna("Mediana")

                    st.dataframe(tabela_exibicao, use_container_width=True, height=360)

    with tab_exportacao:
        st.subheader("Exportação")

        excel_bytes = exportar_excel(
            base_bruta=base_padronizada,
            base_validos=processados,
            inconsistencias=inconsistencias,
            expurgados=expurgados,
            anomalias=anomalias,
            medianas=medianas,
            evolucao_cliente=evolucao_cliente,
        )

        st.download_button(
            label="Baixar Excel consolidado",
            data=excel_bytes.getvalue(),
            file_name=f"{parametros_processados['unidade']}_luna_resultado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
