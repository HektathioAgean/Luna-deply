from __future__ import annotations

import os
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import APP_TITLE, AVAILABLE_UNITS, LAYOUT
from src.data_loader import load_unit_file, list_available_unit_files
from src.data_transformer import aplicar_regras_operacionais, transform_base
from src.engine import (
    build_kpis,
    calcular_medianas_por_cliente,
    excel_export_supported,
    exportar_excel,
    exportar_zip_csv,
)
from src.schema import (
    analyze_schema,
    get_aliases_dataframe,
    get_schema_dataframe,
    schema_report_to_dict,
    standardize_columns,
    suggest_missing_columns,
)
from src.window_calculator import calcular_janelas, resumo_janelas

os.environ["OMP_NUM_THREADS"] = "1"

st.set_page_config(
    page_title=APP_TITLE,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)

STATE_KEYS = [
    "resultado_pronto",
    "ultima_unidade",
    "ultimo_param_hash",
    "base_bruta",
    "base_padronizada",
    "schema_report",
    "relatorio_validacao",
    "base_validos",
    "inconsistencias",
    "processados",
    "expurgados",
    "anomalias",
    "medianas",
    "kpis",
    "janelas",
    "janelas_resumo",
]


def init_state() -> None:
    for key in STATE_KEYS:
        st.session_state.setdefault(key, None)
    if st.session_state["resultado_pronto"] is None:
        st.session_state["resultado_pronto"] = False


# ── Cache helpers ─────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def cached_load_unit_file(unidade: str) -> pd.DataFrame:
    return load_unit_file(unidade)


@st.cache_data(show_spinner=False)
def cached_standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return standardize_columns(df)


@st.cache_data(show_spinner=False)
def cached_transform_base(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return transform_base(df)


@st.cache_data(show_spinner=False)
def cached_aplicar_regras(
    df: pd.DataFrame,
    tempo_min_expurgo: int,
    tempo_max_anomalia: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return aplicar_regras_operacionais(
        df,
        tempo_min_expurgo=tempo_min_expurgo,
        tempo_max_anomalia=tempo_max_anomalia,
    )


@st.cache_data(show_spinner=False)
def cached_calcular_medianas(
    df: pd.DataFrame,
    eventos_previos: int,
    minimo_apontamentos: int,
    tempo_padrao_poucos_apontamentos: int,
    ajuste_percentual: int,
) -> pd.DataFrame:
    return calcular_medianas_por_cliente(
        df=df,
        eventos_previos=eventos_previos,
        minimo_apontamentos=minimo_apontamentos,
        tempo_padrao_poucos_apontamentos=tempo_padrao_poucos_apontamentos,
        ajuste_percentual=ajuste_percentual,
    )


@st.cache_data(show_spinner=False)
def cached_build_kpis(
    base_bruta: pd.DataFrame,
    base_validos: pd.DataFrame,
    inconsistencias: pd.DataFrame,
    expurgados: pd.DataFrame,
    anomalias: pd.DataFrame,
    medianas: pd.DataFrame,
) -> dict:
    return build_kpis(
        base_bruta=base_bruta,
        base_validos=base_validos,
        inconsistencias=inconsistencias,
        expurgados=expurgados,
        anomalias=anomalias,
        medianas=medianas,
    )


@st.cache_data(show_spinner=False)
def cached_calcular_janelas(
    df: pd.DataFrame,
    cobertura_alvo: float,
    min_entregas: int,
) -> pd.DataFrame:
    return calcular_janelas(
        df=df,
        cobertura_alvo=cobertura_alvo,
        min_entregas=min_entregas,
    )


@st.cache_data(show_spinner=False)
def cached_exportar_excel(
    base_bruta: pd.DataFrame,
    base_validos: pd.DataFrame,
    inconsistencias: pd.DataFrame,
    expurgados: pd.DataFrame,
    anomalias: pd.DataFrame,
    medianas: pd.DataFrame,
) -> bytes:
    return exportar_excel(
        base_bruta=base_bruta,
        base_validos=base_validos,
        inconsistencias=inconsistencias,
        expurgados=expurgados,
        anomalias=anomalias,
        medianas=medianas,
    ).getvalue()


@st.cache_data(show_spinner=False)
def cached_exportar_zip_csv(
    base_bruta: pd.DataFrame,
    base_validos: pd.DataFrame,
    inconsistencias: pd.DataFrame,
    expurgados: pd.DataFrame,
    anomalias: pd.DataFrame,
    medianas: pd.DataFrame,
) -> bytes:
    return exportar_zip_csv(
        base_bruta=base_bruta,
        base_validos=base_validos,
        inconsistencias=inconsistencias,
        expurgados=expurgados,
        anomalias=anomalias,
        medianas=medianas,
    ).getvalue()


@st.cache_data(show_spinner=False)
def cached_exportar_medianas_csv_zip(medianas: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    csv_bytes = medianas.to_csv(index=False).encode("utf-8-sig")
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("medianas_cliente.csv", csv_bytes)
    buffer.seek(0)
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def cached_exportar_inconsistencias_csv_zip(inconsistencias: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    csv_bytes = inconsistencias.to_csv(index=False).encode("utf-8-sig")
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("inconsistencias.csv", csv_bytes)
    buffer.seek(0)
    return buffer.getvalue()


def get_params_dict(
    tempo_min_expurgo: int,
    tempo_max_anomalia: int,
    eventos_previos: int,
    minimo_apontamentos: int,
    tempo_padrao_poucos_apontamentos: int,
    ajuste_percentual: int,
) -> dict:
    return {
        "tempo_min_expurgo": int(tempo_min_expurgo),
        "tempo_max_anomalia": int(tempo_max_anomalia),
        "eventos_previos": int(eventos_previos),
        "minimo_apontamentos": int(minimo_apontamentos),
        "tempo_padrao_poucos_apontamentos": int(tempo_padrao_poucos_apontamentos),
        "ajuste_percentual": int(ajuste_percentual),
    }


# ── Processamento principal ──────────────────────────────────────────────────


def processar_analise(unidade: str, params: dict) -> None:
    with st.status("Processando base", expanded=True) as status:
        status.write("Carregando arquivo da unidade")
        base_bruta = cached_load_unit_file(unidade)
        if base_bruta.empty:
            raise ValueError("A base está vazia ou não pôde ser carregada.")
        status.write("Padronizando colunas e validando estrutura")
        base_padronizada = cached_standardize_columns(base_bruta)
        schema_report = analyze_schema(base_padronizada)
        relatorio_validacao = schema_report_to_dict(schema_report)
        if not relatorio_validacao["is_valid"]:
            st.session_state["base_bruta"] = base_bruta
            st.session_state["base_padronizada"] = base_padronizada
            st.session_state["schema_report"] = schema_report
            st.session_state["relatorio_validacao"] = relatorio_validacao
            st.session_state["resultado_pronto"] = False
            status.update(label="Validação reprovada", state="error", expanded=True)
            return
        status.write("Classificando base válida e inconsistências")
        base_validos, inconsistencias = cached_transform_base(base_padronizada)
        status.write("Aplicando expurgo e anomalias")
        processados, expurgados, anomalias = cached_aplicar_regras(
            base_validos,
            tempo_min_expurgo=params["tempo_min_expurgo"],
            tempo_max_anomalia=params["tempo_max_anomalia"],
        )
        status.write("Calculando medianas por cliente")
        medianas = cached_calcular_medianas(
            df=processados,
            eventos_previos=params["eventos_previos"],
            minimo_apontamentos=params["minimo_apontamentos"],
            tempo_padrao_poucos_apontamentos=params["tempo_padrao_poucos_apontamentos"],
            ajuste_percentual=params["ajuste_percentual"],
        )
        status.write("Calculando janelas de entrega")
        janelas = cached_calcular_janelas(
            df=processados,
            cobertura_alvo=0.80,
            min_entregas=params["minimo_apontamentos"],
        )
        janelas_resumo = resumo_janelas(janelas)
        status.write("Consolidando indicadores")
        kpis = cached_build_kpis(
            base_bruta=base_padronizada,
            base_validos=processados,
            inconsistencias=inconsistencias,
            expurgados=expurgados,
            anomalias=anomalias,
            medianas=medianas,
        )
        st.session_state["ultima_unidade"] = unidade
        st.session_state["ultimo_param_hash"] = tuple(sorted(params.items()))
        st.session_state["base_bruta"] = base_bruta
        st.session_state["base_padronizada"] = base_padronizada
        st.session_state["schema_report"] = schema_report
        st.session_state["relatorio_validacao"] = relatorio_validacao
        st.session_state["base_validos"] = base_validos
        st.session_state["inconsistencias"] = inconsistencias
        st.session_state["processados"] = processados
        st.session_state["expurgados"] = expurgados
        st.session_state["anomalias"] = anomalias
        st.session_state["medianas"] = medianas
        st.session_state["kpis"] = kpis
        st.session_state["janelas"] = janelas
        st.session_state["janelas_resumo"] = janelas_resumo
        st.session_state["resultado_pronto"] = True
        status.update(label="Processamento concluído", state="complete", expanded=False)


# ── Renderização ─────────────────────────────────────────────────────────────


def render_empty_state(available_files: list[str]) -> None:
    st.info("Selecione a unidade, ajuste os parâmetros e clique em Processar análise.")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Arquivos disponíveis")
        if available_files:
            st.dataframe(pd.DataFrame({"Arquivo": available_files}), width="stretch", hide_index=True)
        else:
            st.warning("Nenhum arquivo no padrão *_data.xlsx foi encontrado.")
    with col2:
        st.subheader("Fluxo recomendado")
        st.markdown("1. Escolha a unidade.\n2. Revise os parâmetros avançados.\n3. Clique em **Processar análise**.\n4. Use os filtros sem reprocessar a base.\n5. Exporte só o que precisar.")


def render_validation_error(relatorio_validacao: dict, schema_report) -> None:
    st.error("A estrutura da base não atende ao schema mínimo para processamento.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Colunas encontradas", relatorio_validacao["total_columns"])
    c2.metric("Obrigatórias encontradas", len(relatorio_validacao["required_found"]))
    c3.metric("Obrigatórias ausentes", len(relatorio_validacao["required_missing"]))
    st.subheader("Colunas obrigatórias ausentes")
    st.dataframe(pd.DataFrame({"Coluna": relatorio_validacao["required_missing"]}), width="stretch", hide_index=True)
    sugestoes = suggest_missing_columns(schema_report)
    if sugestoes:
        st.subheader("Ações sugeridas")
        for sugestao in sugestoes:
            st.write(f"- {sugestao}")


def render_summary(unidade: str, params: dict, kpis: dict) -> None:
    st.subheader("Resumo executivo")
    a, b, c, d, e, f = st.columns(6)
    a.metric("Linhas válidas", kpis["linhas_validas"])
    b.metric("Clientes únicos", kpis["clientes_unicos"])
    c.metric("Inconsistências", kpis["inconsistencias"])
    d.metric("Expurgados", kpis["expurgados"])
    e.metric("Anomalias", kpis["anomalias"])
    f.metric("Mediana global", kpis["mediana_global_fmt"])


def render_clientes(medianas: pd.DataFrame, processados: pd.DataFrame) -> None:
    st.subheader("Clientes")
    if medianas.empty:
        st.warning("Nenhuma mediana foi calculada para os filtros atuais.")
        return
    clientes = ["Todos"] + sorted(medianas["Cod_Cliente"].astype(str).unique().tolist())
    cliente_selecionado = st.selectbox("Cliente", clientes, index=0)
    med_exibicao = medianas.copy()
    proc_exibicao = processados.copy()
    if cliente_selecionado != "Todos":
        med_exibicao = med_exibicao[med_exibicao["Cod_Cliente"].astype(str) == cliente_selecionado]
        proc_exibicao = proc_exibicao[proc_exibicao["Cod_Cliente"].astype(str) == cliente_selecionado]
    st.dataframe(med_exibicao[["Cod_Cliente", "Qtd_Apontamentos", "Mediana_Tempo_Formatada", "Metodo_Aplicado"]], width="stretch", hide_index=True, height=320)
    if not proc_exibicao.empty:
        base_grafico = proc_exibicao.sort_values("Chegou_em")[["Chegou_em", "Tempo_Sec", "Cod_Cliente"]].copy()
        base_grafico["Tempo_Min"] = base_grafico["Tempo_Sec"] / 60
        st.line_chart(base_grafico.set_index("Chegou_em")["Tempo_Min"], height=260)


def render_qualidade(base_padronizada: pd.DataFrame, relatorio_validacao: dict, inconsistencias: pd.DataFrame) -> None:
    st.subheader("Qualidade da base")
    x1, x2, x3 = st.columns(3)
    x1.metric("Colunas encontradas", relatorio_validacao["total_columns"])
    x2.metric("Extras não reconhecidas", len(relatorio_validacao["unknown_columns"]))
    x3.metric("Linhas com inconsistência", len(inconsistencias))
    with st.expander("Prévia da base padronizada", expanded=False):
        st.dataframe(base_padronizada.head(100), width="stretch", height=360)


def render_detalhes(inconsistencias: pd.DataFrame, expurgados: pd.DataFrame, anomalias: pd.DataFrame, processados: pd.DataFrame) -> None:
    st.subheader("Detalhes operacionais")
    subtabs = st.tabs(["Processados", "Inconsistências", "Expurgados", "Anomalias"])
    with subtabs[0]:
        st.dataframe(processados, width="stretch", height=320, hide_index=True)
    with subtabs[1]:
        st.dataframe(inconsistencias, width="stretch", height=320, hide_index=True)
    with subtabs[2]:
        st.dataframe(expurgados, width="stretch", height=320, hide_index=True)
    with subtabs[3]:
        st.dataframe(anomalias, width="stretch", height=320, hide_index=True)


def render_janelas(janelas: pd.DataFrame, janelas_resumo: dict) -> None:
    """Aba de janelas operacionais de entrega por cliente."""
    st.subheader("Janelas de entrega")

    if janelas is None or janelas.empty:
        st.warning("Nenhuma janela calculada. Verifique se há dados processados suficientes.")
        return

    # ── KPIs ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Clientes com janela", janelas_resumo["clientes_com_janela"])
    k2.metric("Amplitude média", f'{janelas_resumo["amplitude_media_min"]} min')
    k3.metric("Cobertura média", f'{janelas_resumo["cobertura_media"]}%')
    k4.metric("Período dominante", janelas_resumo["periodo_mais_comum"])
    k5.metric("Cruzam meia-noite", janelas_resumo["clientes_meianoite"])

    # ── Filtros (sem reprocessar) ─────────────────────────────────────────
    col_filtro1, col_filtro2 = st.columns([1, 1])
    with col_filtro1:
        periodos_disponiveis = ["Todos"] + sorted(
            janelas.dropna(subset=["Amplitude_Min"])["Periodo_Pico"].unique().tolist()
        )
        filtro_periodo = st.selectbox(
            "Filtrar por período", periodos_disponiveis, index=0, key="janela_periodo"
        )
    with col_filtro2:
        max_clientes = st.slider(
            "Clientes no gráfico", min_value=5, max_value=80, value=30, step=5, key="janela_max"
        )

    # Aplicar filtro
    df_vis = janelas.dropna(subset=["Amplitude_Min"]).copy()
    if filtro_periodo != "Todos":
        df_vis = df_vis[df_vis["Periodo_Pico"] == filtro_periodo]

    if df_vis.empty:
        st.info("Nenhum cliente com janela válida neste filtro.")
        return

    df_vis = df_vis.head(max_clientes)

    # ── Gráfico horizontal de janelas ─────────────────────────────────────
    # Montar barras como gráfico Streamlit nativo via Altair/Vega
    # Cada cliente é uma barra horizontal do Inicio_Min ao Fim_Min

    chart_data = []
    for _, row in df_vis.iterrows():
        inicio = row["Inicio_Min"]
        fim = row["Fim_Min"]
        cruza = row["Cruza_MeiaNoite"]
        cliente_label = str(row["Cod_Cliente"])

        if cruza:
            # Duas barras: inicio→1440 e 0→fim
            chart_data.append({
                "Cliente": cliente_label,
                "Inicio": inicio,
                "Fim": 1440,
                "Janela": f'{row["Janela_Inicio"]} – {row["Janela_Fim"]}',
                "Tipo": "Cruza meia-noite",
            })
            chart_data.append({
                "Cliente": cliente_label,
                "Inicio": 0,
                "Fim": fim,
                "Janela": f'{row["Janela_Inicio"]} – {row["Janela_Fim"]}',
                "Tipo": "Cruza meia-noite",
            })
        else:
            chart_data.append({
                "Cliente": cliente_label,
                "Inicio": inicio,
                "Fim": fim,
                "Janela": f'{row["Janela_Inicio"]} – {row["Janela_Fim"]}',
                "Tipo": row["Periodo_Pico"],
            })

    df_chart = pd.DataFrame(chart_data)

    # Usar st.bar_chart não suporta barras horizontais com range,
    # então construímos com o Vega-Lite embutido do Streamlit.
    cores_periodo = {
        "Manhã": "#1D9E75",
        "Tarde": "#378ADD",
        "Noite": "#7F77DD",
        "Madrugada": "#D85A30",
        "Cruza meia-noite": "#EF9F27",
    }

    # Montar horas para eixo X
    tick_values = [h * 60 for h in range(0, 25, 2)]
    tick_labels = [f"{h:02d}:00" for h in range(0, 25, 2)]

    vega_spec = {
        "data": {"values": df_chart.to_dict(orient="records")},
        "mark": {
            "type": "bar",
            "cornerRadiusEnd": 3,
            "height": {"band": 0.7},
        },
        "encoding": {
            "y": {
                "field": "Cliente",
                "type": "nominal",
                "sort": None,
                "axis": {"title": None, "labelFontSize": 11},
            },
            "x": {
                "field": "Inicio",
                "type": "quantitative",
                "scale": {"domain": [0, 1440]},
                "axis": {
                    "title": "Horário",
                    "values": tick_values,
                    "labelExpr": "floor(datum.value / 60) + ':' + (datum.value % 60 < 10 ? '0' : '') + (datum.value % 60)",
                    "labelFontSize": 11,
                },
            },
            "x2": {"field": "Fim"},
            "color": {
                "field": "Tipo",
                "type": "nominal",
                "scale": {
                    "domain": list(cores_periodo.keys()),
                    "range": list(cores_periodo.values()),
                },
                "legend": {"title": "Período", "orient": "top"},
            },
            "tooltip": [
                {"field": "Cliente", "type": "nominal"},
                {"field": "Janela", "type": "nominal", "title": "Janela"},
                {"field": "Tipo", "type": "nominal", "title": "Período"},
            ],
        },
        "height": max(len(df_vis) * 24, 200),
        "width": "container",
    }

    st.vega_lite_chart(vega_spec, width="stretch")

    # ── Tabela de dados ───────────────────────────────────────────────────
    with st.expander("Tabela de janelas por cliente", expanded=False):
        colunas_exibir = [
            "Cod_Cliente", "Qtd_Entregas", "Janela_Inicio", "Janela_Fim",
            "Amplitude_Min", "Horario_Pico", "Periodo_Pico",
            "Cobertura_Efetiva", "Cruza_MeiaNoite",
        ]
        colunas_presentes = [c for c in colunas_exibir if c in df_vis.columns]
        st.dataframe(
            df_vis[colunas_presentes],
            width="stretch",
            hide_index=True,
            height=400,
        )

    # ── Download janelas CSV ──────────────────────────────────────────────
    csv_janelas = janelas.dropna(subset=["Amplitude_Min"]).to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Baixar janelas em CSV",
        csv_janelas,
        f"{st.session_state.get('ultima_unidade', 'unidade')}_janelas.csv",
        "text/csv",
        width="stretch",
    )


def render_exportacao(unidade: str, base_padronizada: pd.DataFrame, processados: pd.DataFrame, inconsistencias: pd.DataFrame, expurgados: pd.DataFrame, anomalias: pd.DataFrame, medianas: pd.DataFrame) -> None:
    st.subheader("Exportação")
    pacote_csv_zip = cached_exportar_zip_csv(base_padronizada, processados, inconsistencias, expurgados, anomalias, medianas)
    medianas_zip = cached_exportar_medianas_csv_zip(medianas)
    inconsistencias_zip = cached_exportar_inconsistencias_csv_zip(inconsistencias)
    excel_bytes = None
    excel_error = None

    if excel_export_supported():
        try:
            excel_bytes = cached_exportar_excel(base_padronizada, processados, inconsistencias, expurgados, anomalias, medianas)
        except Exception as exc:
            excel_error = str(exc)
    else:
        excel_error = "A exportacao em Excel esta indisponivel neste ambiente porque o pacote 'openpyxl' nao esta instalado."

    if excel_error:
        st.warning(f"{excel_error} O pacote CSV completo continua disponível.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if excel_bytes is not None:
            st.download_button("Baixar Excel completo", excel_bytes, f"{unidade}_luna_resultado.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
        else:
            st.button("Excel indisponível", disabled=True, width="stretch")
    with c2:
        st.download_button("Baixar pacote CSV", pacote_csv_zip, f"{unidade}_luna_resultado_csv.zip", "application/zip", width="stretch")
    with c3:
        st.download_button("Baixar medianas em ZIP", medianas_zip, f"{unidade}_medianas.zip", "application/zip", width="stretch")
    with c4:
        st.download_button("Baixar inconsistências em ZIP", inconsistencias_zip, f"{unidade}_inconsistencias.zip", "application/zip", width="stretch")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    init_state()
    st.title("Luna")
    st.caption("Análise de tempos operacionais com foco em decisão e validação rápida.")
    available_files = list_available_unit_files()
    with st.sidebar:
        st.header("Configuração")
        with st.form("form_processamento"):
            unidade = st.selectbox("Unidade", options=AVAILABLE_UNITS, index=0)
            with st.expander("Parâmetros avançados", expanded=False):
                tempo_min_expurgo = st.number_input("Tempo mínimo de expurgo (segundos)", min_value=1, value=300, step=1)
                tempo_max_anomalia = st.number_input("Tempo máximo para anomalia (segundos)", min_value=1, value=21600, step=1)
                eventos_previos = st.number_input("Eventos prévios", min_value=1, value=10, step=1)
                minimo_apontamentos = st.number_input("Mínimo de apontamentos por cliente", min_value=1, value=4, step=1)
                tempo_padrao_poucos_apontamentos = st.number_input("Tempo padrão para poucos apontamentos (segundos)", min_value=1, value=600, step=1)
                ajuste_percentual = st.slider("Ajuste percentual", min_value=-20, max_value=100, value=0, step=1)
            processar = st.form_submit_button("Processar análise", width="stretch")
        st.divider()
        st.caption("Arquivos detectados")
        if available_files:
            st.write(available_files)
        else:
            st.warning("Nenhum arquivo no padrão *_data.xlsx foi encontrado.")
    params = get_params_dict(tempo_min_expurgo, tempo_max_anomalia, eventos_previos, minimo_apontamentos, tempo_padrao_poucos_apontamentos, ajuste_percentual)
    if processar:
        try:
            processar_analise(unidade, params)
        except Exception as exc:
            st.error(str(exc))
            st.stop()
    relatorio_validacao = st.session_state.get("relatorio_validacao")
    if relatorio_validacao and not relatorio_validacao["is_valid"]:
        render_validation_error(relatorio_validacao, st.session_state.get("schema_report"))
        return
    if not st.session_state.get("resultado_pronto"):
        render_empty_state(available_files)
        return
    tab_resumo, tab_clientes, tab_janelas, tab_qualidade, tab_detalhes, tab_exportacao = st.tabs(
        ["Resumo", "Clientes", "Janelas", "Qualidade da base", "Detalhes", "Exportação"]
    )
    with tab_resumo:
        render_summary(st.session_state["ultima_unidade"], params, st.session_state["kpis"])
    with tab_clientes:
        render_clientes(st.session_state["medianas"], st.session_state["processados"])
    with tab_janelas:
        render_janelas(st.session_state["janelas"], st.session_state["janelas_resumo"])
    with tab_qualidade:
        render_qualidade(st.session_state["base_padronizada"], relatorio_validacao, st.session_state["inconsistencias"])
    with tab_detalhes:
        render_detalhes(st.session_state["inconsistencias"], st.session_state["expurgados"], st.session_state["anomalias"], st.session_state["processados"])
    with tab_exportacao:
        render_exportacao(st.session_state["ultima_unidade"], st.session_state["base_padronizada"], st.session_state["processados"], st.session_state["inconsistencias"], st.session_state["expurgados"], st.session_state["anomalias"], st.session_state["medianas"])


if __name__ == "__main__":
    main()
