from __future__ import annotations

import os
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import APP_TITLE, AVAILABLE_UNITS, LAYOUT
from src.data_loader import load_unit_file, list_available_unit_files
from src.data_transformer import aplicar_regras_operacionais, transform_base
from src.engine import build_kpis, calcular_medianas_por_cliente, exportar_excel
from src.schema import (
    analyze_schema,
    schema_report_to_dict,
    standardize_columns,
    suggest_missing_columns,
)
from src.window_calculator import calcular_janelas, resumo_janelas

os.environ["OMP_NUM_THREADS"] = "1"

st.set_page_config(page_title=APP_TITLE, layout=LAYOUT, initial_sidebar_state="expanded")


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — única fonte de verdade após o processamento
# ══════════════════════════════════════════════════════════════════════════════

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
    "lista_clientes",
    "proc_cliente_str",
    "export_excel_bytes",
    "export_medianas_zip",
    "export_inconsistencias_zip",
]


def init_state() -> None:
    for key in STATE_KEYS:
        st.session_state.setdefault(key, None)
    if st.session_state["resultado_pronto"] is None:
        st.session_state["resultado_pronto"] = False


# ══════════════════════════════════════════════════════════════════════════════
# CACHE — apenas na leitura do arquivo (chave = string, leve)
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data(show_spinner=False)
def cached_load_unit_file(unidade: str) -> pd.DataFrame:
    """Único ponto de cache real — chave é a string da unidade, sem hash de DF."""
    return load_unit_file(unidade)


# ══════════════════════════════════════════════════════════════════════════════
# PROCESSAMENTO — roda 1 vez ao clicar "Processar", grava tudo no state
# ══════════════════════════════════════════════════════════════════════════════


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


def processar_analise(unidade: str, params: dict) -> None:
    """Pipeline completo — executa tudo e salva no session_state."""
    param_hash = tuple(sorted(params.items()))

    if (
        st.session_state.get("resultado_pronto")
        and st.session_state.get("ultima_unidade") == unidade
        and st.session_state.get("ultimo_param_hash") == param_hash
    ):
        return

    with st.status("Processando base", expanded=True) as status:
        status.write("Carregando arquivo da unidade…")
        base_bruta = cached_load_unit_file(unidade)
        if base_bruta.empty:
            raise ValueError("A base está vazia ou não pôde ser carregada.")

        status.write("Padronizando colunas…")
        base_padronizada = standardize_columns(base_bruta)
        schema_report = analyze_schema(base_padronizada)
        relatorio_validacao = schema_report_to_dict(schema_report)

        if not relatorio_validacao["is_valid"]:
            st.session_state.update(
                base_bruta=base_bruta,
                base_padronizada=base_padronizada,
                schema_report=schema_report,
                relatorio_validacao=relatorio_validacao,
                resultado_pronto=False,
            )
            status.update(label="Validação reprovada", state="error", expanded=True)
            return

        status.write("Classificando válidos e inconsistências…")
        base_validos, inconsistencias = transform_base(base_padronizada)

        status.write("Aplicando expurgo e anomalias…")
        processados, expurgados, anomalias = aplicar_regras_operacionais(
            base_validos,
            tempo_min_expurgo=params["tempo_min_expurgo"],
            tempo_max_anomalia=params["tempo_max_anomalia"],
        )

        status.write("Calculando medianas por cliente…")
        medianas = calcular_medianas_por_cliente(
            df=processados,
            eventos_previos=params["eventos_previos"],
            minimo_apontamentos=params["minimo_apontamentos"],
            tempo_padrao_poucos_apontamentos=params["tempo_padrao_poucos_apontamentos"],
            ajuste_percentual=params["ajuste_percentual"],
        )

        status.write("Calculando janelas de entrega…")
        janelas = calcular_janelas(
            df=processados,
            cobertura_alvo=0.80,
            min_entregas=params["minimo_apontamentos"],
        )
        janelas_res = resumo_janelas(janelas)

        status.write("Consolidando indicadores…")
        kpis = build_kpis(
            base_bruta=base_padronizada,
            base_validos=processados,
            inconsistencias=inconsistencias,
            expurgados=expurgados,
            anomalias=anomalias,
            medianas=medianas,
        )

        proc_cliente_str = processados["Cod_Cliente"].astype(str)
        lista_clientes = sorted(proc_cliente_str.unique().tolist())

        st.session_state.update(
            ultima_unidade=unidade,
            ultimo_param_hash=param_hash,
            base_bruta=base_bruta,
            base_padronizada=base_padronizada,
            schema_report=schema_report,
            relatorio_validacao=relatorio_validacao,
            base_validos=base_validos,
            inconsistencias=inconsistencias,
            processados=processados,
            expurgados=expurgados,
            anomalias=anomalias,
            medianas=medianas,
            kpis=kpis,
            janelas=janelas,
            janelas_resumo=janelas_res,
            lista_clientes=lista_clientes,
            proc_cliente_str=proc_cliente_str,
            export_excel_bytes=None,
            export_medianas_zip=None,
            export_inconsistencias_zip=None,
            resultado_pronto=True,
        )
        status.update(label="Processamento concluído", state="complete", expanded=False)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTAÇÃO — gerada sob demanda (lazy), guardada no state
# ══════════════════════════════════════════════════════════════════════════════


def _get_export_excel() -> bytes:
    if st.session_state["export_excel_bytes"] is None:
        st.session_state["export_excel_bytes"] = exportar_excel(
            base_bruta=st.session_state["base_padronizada"],
            base_validos=st.session_state["processados"],
            inconsistencias=st.session_state["inconsistencias"],
            expurgados=st.session_state["expurgados"],
            anomalias=st.session_state["anomalias"],
            medianas=st.session_state["medianas"],
        ).getvalue()
    return st.session_state["export_excel_bytes"]


def _get_export_medianas_zip() -> bytes:
    if st.session_state["export_medianas_zip"] is None:
        buffer = BytesIO()
        csv_bytes = st.session_state["medianas"].to_csv(index=False).encode("utf-8-sig")
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("medianas_cliente.csv", csv_bytes)
        buffer.seek(0)
        st.session_state["export_medianas_zip"] = buffer.getvalue()
    return st.session_state["export_medianas_zip"]


def _get_export_inconsistencias_zip() -> bytes:
    if st.session_state["export_inconsistencias_zip"] is None:
        buffer = BytesIO()
        csv_bytes = st.session_state["inconsistencias"].to_csv(index=False).encode("utf-8-sig")
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("inconsistencias.csv", csv_bytes)
        buffer.seek(0)
        st.session_state["export_inconsistencias_zip"] = buffer.getvalue()
    return st.session_state["export_inconsistencias_zip"]


# ══════════════════════════════════════════════════════════════════════════════
# RENDERIZAÇÃO — lê do session_state, sem hash, sem cache, sem cópia
# ══════════════════════════════════════════════════════════════════════════════


def render_empty_state(available_files: list[str]) -> None:
    st.info("Selecione a unidade, ajuste os parâmetros e clique em Processar análise.")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Arquivos disponíveis")
        if available_files:
            st.dataframe(
                pd.DataFrame({"Arquivo": available_files}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("Nenhum arquivo no padrão *_data.xlsx foi encontrado.")
    with col2:
        st.subheader("Fluxo recomendado")
        st.markdown(
            "1. Escolha a unidade.\n"
            "2. Revise os parâmetros avançados.\n"
            "3. Clique em **Processar análise**.\n"
            "4. Use os filtros sem reprocessar a base.\n"
            "5. Exporte só o que precisar."
        )


def render_validation_error() -> None:
    rv = st.session_state["relatorio_validacao"]
    st.error("A estrutura da base não atende ao schema mínimo para processamento.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Colunas encontradas", rv["total_columns"])
    c2.metric("Obrigatórias encontradas", len(rv["required_found"]))
    c3.metric("Obrigatórias ausentes", len(rv["required_missing"]))
    st.subheader("Colunas obrigatórias ausentes")
    st.dataframe(
        pd.DataFrame({"Coluna": rv["required_missing"]}),
        use_container_width=True,
        hide_index=True,
    )
    sugestoes = suggest_missing_columns(st.session_state["schema_report"])
    if sugestoes:
        st.subheader("Ações sugeridas")
        for sugestao in sugestoes:
            st.write(f"- {sugestao}")


def render_summary() -> None:
    kpis = st.session_state["kpis"]
    st.subheader("Resumo executivo")
    a, b, c, d, e, f = st.columns(6)
    a.metric("Linhas válidas", kpis["linhas_validas"])
    b.metric("Clientes únicos", kpis["clientes_unicos"])
    c.metric("Inconsistências", kpis["inconsistencias"])
    d.metric("Expurgados", kpis["expurgados"])
    e.metric("Anomalias", kpis["anomalias"])
    f.metric("Mediana global", kpis["mediana_global_fmt"])


def render_clientes() -> None:
    medianas = st.session_state["medianas"]
    processados = st.session_state["processados"]
    proc_str = st.session_state["proc_cliente_str"]

    st.subheader("Clientes")
    if medianas.empty:
        st.warning("Nenhuma mediana foi calculada para os filtros atuais.")
        return

    clientes = ["Todos"] + st.session_state["lista_clientes"]
    cliente_sel = st.selectbox("Cliente", clientes, index=0, key="tab_clientes_sel")

    if cliente_sel == "Todos":
        med_exib = medianas
        proc_exib = processados
    else:
        mask_med = medianas["Cod_Cliente"].astype(str) == cliente_sel
        mask_proc = proc_str == cliente_sel
        med_exib = medianas.loc[mask_med]
        proc_exib = processados.loc[mask_proc]

    st.dataframe(
        med_exib[["Cod_Cliente", "Qtd_Apontamentos", "Mediana_Tempo_Formatada", "Metodo_Aplicado"]],
        use_container_width=True,
        hide_index=True,
        height=320,
    )

    if not proc_exib.empty:
        base_g = proc_exib.sort_values("Chegou_em")[["Chegou_em", "Tempo_Sec"]].copy()
        base_g["Tempo_Min"] = base_g["Tempo_Sec"] / 60
        st.line_chart(base_g.set_index("Chegou_em")["Tempo_Min"], height=260)


def render_janelas() -> None:
    """Aba Janelas — 1 cliente por vez, scatter de horários + tempo+volume."""
    janelas = st.session_state["janelas"]
    janelas_resumo = st.session_state["janelas_resumo"]
    processados = st.session_state["processados"]
    proc_str = st.session_state["proc_cliente_str"]

    st.subheader("Janelas de entrega")

    if janelas is None or janelas.empty:
        st.warning("Nenhuma janela calculada.")
        return

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Clientes com janela", janelas_resumo["clientes_com_janela"])
    k2.metric("Amplitude média", f'{janelas_resumo["amplitude_media_min"]} min')
    k3.metric("Cobertura média", f'{janelas_resumo["cobertura_media"]}%')
    k4.metric("Período dominante", janelas_resumo["periodo_mais_comum"])
    k5.metric("Cruzam meia-noite", janelas_resumo["clientes_meianoite"])

    st.divider()

    lista_janela = sorted(
        janelas.dropna(subset=["Amplitude_Min"])["Cod_Cliente"].astype(str).unique().tolist()
    )
    if not lista_janela:
        st.info("Nenhum cliente com janela válida.")
        return

    cliente_sel = st.selectbox(
        "Selecione o cliente (digite para buscar)",
        options=lista_janela,
        index=0,
        key="janela_cliente_sel",
    )

    mask = proc_str == cliente_sel
    df_cli = processados.loc[mask]

    if df_cli.empty:
        st.warning(f"Nenhum registro processado para {cliente_sel}.")
        return

    ij_row = janelas[janelas["Cod_Cliente"].astype(str) == cliente_sel]
    ij = ij_row.iloc[0] if not ij_row.empty else None

    if ij is not None:
        j1, j2, j3, j4 = st.columns(4)
        j1.metric("Janela", f'{ij["Janela_Inicio"]} – {ij["Janela_Fim"]}')
        j2.metric("Amplitude", f'{ij["Amplitude_Min"]} min')
        j3.metric("Cobertura", f'{ij["Cobertura_Efetiva"]}%')
        j4.metric("Entregas", ij["Qtd_Entregas"])

    chegada_dt = pd.to_datetime(df_cli["Chegou_em"], errors="coerce")
    data_str = chegada_dt.dt.strftime("%Y-%m-%d").values
    hora_min = (chegada_dt.dt.hour * 60 + chegada_dt.dt.minute).values

    st.markdown("**Horários de chegada por dia**")

    tick_horas = [h * 60 for h in range(0, 25, 2)]
    layers = []

    if ij is not None:
        rect_spec = {"mark": {"type": "rect", "opacity": 0.10, "color": "#1D9E75"}}
        if ij["Cruza_MeiaNoite"]:
            layers.append(
                {**rect_spec, "encoding": {"y": {"datum": ij["Inicio_Min"]}, "y2": {"datum": 1440}}}
            )
            layers.append(
                {**rect_spec, "encoding": {"y": {"datum": 0}, "y2": {"datum": ij["Fim_Min"]}}}
            )
        else:
            layers.append(
                {**rect_spec, "encoding": {"y": {"datum": ij["Inicio_Min"]}, "y2": {"datum": ij["Fim_Min"]}}}
            )

    scatter_data = [{"d": d, "h": int(h)} for d, h in zip(data_str, hora_min) if pd.notna(h)]

    layers.append(
        {
            "mark": {"type": "circle", "size": 35, "opacity": 0.75},
            "encoding": {
                "x": {
                    "field": "d",
                    "type": "ordinal",
                    "axis": {"title": "Dia", "labelAngle": -45, "labelFontSize": 10},
                },
                "y": {
                    "field": "h",
                    "type": "quantitative",
                    "scale": {"domain": [0, 1440]},
                    "axis": {
                        "title": "Horário",
                        "values": tick_horas,
                        "labelExpr": (
                            "floor(datum.value / 60) + ':' + "
                            "(datum.value % 60 < 10 ? '0' : '') + (datum.value % 60)"
                        ),
                    },
                },
                "color": {"value": "#378ADD"},
                "tooltip": [{"field": "d", "title": "Data"}, {"field": "h", "title": "Min. do dia"}],
            },
        }
    )

    st.vega_lite_chart(
        {"data": {"values": scatter_data}, "layer": layers, "height": 280, "width": "container"},
        use_container_width=True,
    )

    st.markdown("**Tempo de entrega e volume por dia**")

    tem_vol = "Vol_caixas" in df_cli.columns and df_cli["Vol_caixas"].notna().any()

    df_g = df_cli.assign(_data_str=data_str)
    agg = {"Tempo_Sec": "median"}
    if tem_vol:
        agg["Vol_caixas"] = "sum"

    df_dia = df_g.groupby("_data_str").agg(agg).reset_index()
    df_dia["Tempo_Min"] = (df_dia["Tempo_Sec"] / 60).round(1)

    if not tem_vol:
        contagem = df_g.groupby("_data_str").size().reset_index(name="Qtd")
        df_dia = df_dia.merge(contagem, on="_data_str", how="left")

    col_vol = "Vol_caixas" if tem_vol else "Qtd"
    label_vol = "Volume (caixas)" if tem_vol else "Qtd entregas"

    combo_data = (
        df_dia[["_data_str", "Tempo_Min", col_vol]]
        .rename(columns={"_data_str": "d"})
        .to_dict(orient="records")
    )

    st.vega_lite_chart(
        {
            "data": {"values": combo_data},
            "resolve": {"scale": {"y": "independent"}},
            "layer": [
                {
                    "mark": {
                        "type": "bar",
                        "opacity": 0.4,
                        "color": "#85B7EB",
                        "cornerRadiusEnd": 2,
                    },
                    "encoding": {
                        "x": {
                            "field": "d",
                            "type": "ordinal",
                            "axis": {"title": "Dia", "labelAngle": -45, "labelFontSize": 10},
                        },
                        "y": {
                            "field": col_vol,
                            "type": "quantitative",
                            "axis": {"title": label_vol, "titleColor": "#85B7EB"},
                        },
                        "tooltip": [{"field": "d", "title": "Data"}, {"field": col_vol, "title": label_vol}],
                    },
                },
                {
                    "mark": {
                        "type": "line",
                        "color": "#D85A30",
                        "strokeWidth": 2,
                        "point": {"size": 28, "color": "#D85A30"},
                    },
                    "encoding": {
                        "x": {"field": "d", "type": "ordinal"},
                        "y": {
                            "field": "Tempo_Min",
                            "type": "quantitative",
                            "axis": {"title": "Tempo (min)", "titleColor": "#D85A30"},
                        },
                        "tooltip": [{"field": "d", "title": "Data"}, {"field": "Tempo_Min", "title": "Tempo (min)"}],
                    },
                },
            ],
            "height": 280,
            "width": "container",
        },
        use_container_width=True,
    )

    with st.expander("Entregas do cliente", expanded=False):
        cols = [c for c in ["Chegou_em", "Finalizada_em", "Tempo_Sec", "Vol_caixas"] if c in df_cli.columns]
        st.dataframe(df_cli[cols], use_container_width=True, hide_index=True, height=300)

    csv_j = janelas.dropna(subset=["Amplitude_Min"]).to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Baixar todas as janelas em CSV",
        csv_j,
        f"{st.session_state.get('ultima_unidade', 'unidade')}_janelas.csv",
        "text/csv",
        use_container_width=True,
    )


def render_qualidade() -> None:
    rv = st.session_state["relatorio_validacao"]
    inconsistencias = st.session_state["inconsistencias"]
    st.subheader("Qualidade da base")
    x1, x2, x3 = st.columns(3)
    x1.metric("Colunas encontradas", rv["total_columns"])
    x2.metric("Extras não reconhecidas", len(rv["unknown_columns"]))
    x3.metric("Linhas com inconsistência", len(inconsistencias))
    with st.expander("Prévia da base padronizada", expanded=False):
        st.dataframe(st.session_state["base_padronizada"].head(100), use_container_width=True, height=360)


def render_detalhes() -> None:
    st.subheader("Detalhes operacionais")
    subtabs = st.tabs(["Processados", "Inconsistências", "Expurgados", "Anomalias"])
    with subtabs[0]:
        st.dataframe(st.session_state["processados"], use_container_width=True, height=320, hide_index=True)
    with subtabs[1]:
        st.dataframe(st.session_state["inconsistencias"], use_container_width=True, height=320, hide_index=True)
    with subtabs[2]:
        st.dataframe(st.session_state["expurgados"], use_container_width=True, height=320, hide_index=True)
    with subtabs[3]:
        st.dataframe(st.session_state["anomalias"], use_container_width=True, height=320, hide_index=True)


def render_exportacao() -> None:
    unidade = st.session_state["ultima_unidade"]
    st.subheader("Exportação")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Baixar Excel completo",
            _get_export_excel(),
            f"{unidade}_luna_resultado.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Baixar medianas em ZIP",
            _get_export_medianas_zip(),
            f"{unidade}_medianas.zip",
            "application/zip",
            use_container_width=True,
        )
    with c3:
        st.download_button(
            "Baixar inconsistências em ZIP",
            _get_export_inconsistencias_zip(),
            f"{unidade}_inconsistencias.zip",
            "application/zip",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


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
                tempo_min_expurgo = st.number_input(
                    "Tempo mínimo de expurgo (s)",
                    min_value=1,
                    value=300,
                    step=1,
                )
                tempo_max_anomalia = st.number_input(
                    "Tempo máximo para anomalia (s)",
                    min_value=1,
                    value=21600,
                    step=1,
                )
                eventos_previos = st.number_input("Eventos prévios", min_value=1, value=10, step=1)
                minimo_apontamentos = st.number_input(
                    "Mínimo de apontamentos",
                    min_value=1,
                    value=4,
                    step=1,
                )
                tempo_padrao_poucos = st.number_input(
                    "Tempo padrão poucos apontam. (s)",
                    min_value=1,
                    value=600,
                    step=1,
                )
                ajuste_pct = st.slider("Ajuste percentual", min_value=-20, max_value=100, value=0, step=1)
            processar = st.form_submit_button("Processar análise", use_container_width=True)
        st.divider()
        st.caption("Arquivos detectados")
        if available_files:
            st.write(available_files)
        else:
            st.warning("Nenhum arquivo *_data.xlsx encontrado.")

    params = get_params_dict(
        tempo_min_expurgo,
        tempo_max_anomalia,
        eventos_previos,
        minimo_apontamentos,
        tempo_padrao_poucos,
        ajuste_pct,
    )

    if processar:
        try:
            processar_analise(unidade, params)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    rv = st.session_state.get("relatorio_validacao")
    if rv and not rv["is_valid"]:
        render_validation_error()
        return

    if not st.session_state.get("resultado_pronto"):
        render_empty_state(available_files)
        return

    tab_resumo, tab_clientes, tab_janelas, tab_qualidade, tab_detalhes, tab_export = st.tabs(
        ["Resumo", "Clientes", "Janelas", "Qualidade da base", "Detalhes", "Exportação"]
    )
    with tab_resumo:
        render_summary()
    with tab_clientes:
        render_clientes()
    with tab_janelas:
        render_janelas()
    with tab_qualidade:
        render_qualidade()
    with tab_detalhes:
        render_detalhes()
    with tab_export:
        render_exportacao()


if __name__ == "__main__":
    main()
