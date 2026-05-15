import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import APP_TITLE, AVAILABLE_UNITS, LAYOUT
from src.data_loader import list_available_unit_files, load_unit_file
from src.data_transformer import aplicar_regras_operacionais, transform_base
from src.engine import (
    build_kpis,
    calcular_boxplot_iqr_tempos,
    calcular_medianas_por_cliente,
    exportar_excel,
    exportar_zip_csv,
    format_seconds,
)
from src.schema import (
    analyze_schema,
    get_aliases_dataframe,
    get_schema_dataframe,
    schema_report_to_dict,
    standardize_columns,
    suggest_missing_columns,
)
from src.utils import (
    DIAS_SEMANA_MAP,
    OPCOES_DIA_SEMANA,
    calcular_janela_circular_minima,
    classificar_comercial,
    classificar_periodo,
    formatar_minutos_hhmm,
    formatar_numero,
    minutos_desde_meia_noite,
    normalizar_volume_caixas,
    preparar_dataframe_streamlit,
)

os.environ["OMP_NUM_THREADS"] = "1"

st.set_page_config(
    page_title=APP_TITLE,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)


# -- Pipeline principal --------------------------------------------------------

@st.cache_data(show_spinner=False)
def processar_base(
    unidade: str,
    tempo_min_expurgo: int,
    tempo_max_anomalia: int,
    eventos_previos: int,
    minimo_apontamentos: int,
    tempo_padrao_poucos_apontamentos: int,
    ajuste_percentual: float,
) -> dict:
    base_bruta = load_unit_file(unidade)

    if base_bruta.empty:
        raise ValueError("A base esta vazia ou nao pode ser carregada.")

    base_padronizada = standardize_columns(base_bruta)
    schema_report = analyze_schema(base_padronizada)
    relatorio_validacao = schema_report_to_dict(schema_report)

    if not relatorio_validacao["is_valid"]:
        return {
            "base_bruta": base_bruta,
            "base_padronizada": base_padronizada,
            "schema_report": schema_report,
            "relatorio_validacao": relatorio_validacao,
            "inconsistencias": pd.DataFrame(),
            "processados": pd.DataFrame(),
            "expurgados": pd.DataFrame(),
            "anomalias": pd.DataFrame(),
            "medianas": pd.DataFrame(),
            "kpis": {},
        }

    processados_base, inconsistencias = transform_base(base_padronizada)

    processados, expurgados, anomalias = aplicar_regras_operacionais(
        processados_base,
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
    }


# -- Base detalhada: processados + referencias por linha ----------------------

@st.cache_data(show_spinner=False)
def montar_base_detalhada(
    processados: pd.DataFrame,
    medianas: pd.DataFrame,
) -> pd.DataFrame:
    """
    Faz join de processados com as colunas de referencia de medianas por cliente.

    Cada linha da base processada recebe:
    - Mediana_Tempo_Formatada
    - Mediana_Tempo_Sec
    - Tempo_Ideal_Q1_Formatado
    - Tempo_Ideal_Q1_Sec
    - Diferenca_Mediana_Q1_Formatada
    - Diferenca_Mediana_Q1_Percentual
    - Metodo_Aplicado
    - Metodo_Ideal_Aplicado

    Permite analise por entrega individual com referencia do cliente ao lado.
    """
    if processados is None or processados.empty:
        return pd.DataFrame()

    if medianas is None or medianas.empty:
        return processados.copy()

    colunas_referencia = [
        "Cod_Cliente",
        "Mediana_Tempo_Sec",
        "Mediana_Tempo_Formatada",
        "Tempo_Ideal_Q1_Sec",
        "Tempo_Ideal_Q1_Formatado",
        "Diferenca_Mediana_Q1_Sec",
        "Diferenca_Mediana_Q1_Formatada",
        "Diferenca_Mediana_Q1_Percentual",
        "Metodo_Aplicado",
        "Metodo_Ideal_Aplicado",
    ]

    colunas_existentes = [c for c in colunas_referencia if c in medianas.columns]
    medianas_ref = medianas[colunas_existentes].copy()
    medianas_ref["Cod_Cliente"] = medianas_ref["Cod_Cliente"].astype(str).str.strip()

    base = processados.copy()
    base["Cod_Cliente"] = base["Cod_Cliente"].astype(str).str.strip()

    return base.merge(medianas_ref, on="Cod_Cliente", how="left")


# -- Tempos de referencia (recorte local do painel) ----------------------------

def calcular_tempos_referencia_local(serie_tempos: pd.Series) -> dict:
    """
    Calcula mediana e tempo ideal Q1 para o recorte exibido no painel do cliente.
    Usa o mesmo calcular_boxplot_iqr_tempos do engine para consistencia.
    """
    tempos = pd.to_numeric(serie_tempos, errors="coerce").dropna()

    if tempos.empty:
        return {
            "mediana_tempo_sec": 0.0,
            "mediana_tempo_fmt": "00:00:00",
            "tempo_ideal_q1_sec": 0.0,
            "tempo_ideal_q1_fmt": "00:00:00",
            "gap_mediana_q1_sec": 0.0,
            "gap_mediana_q1_fmt": "00:00:00",
            "gap_mediana_q1_perc": 0.0,
            "qtd_outliers_boxplot": 0,
            "qtd_base_limpa_boxplot": 0,
            "metodo_ideal_aplicado": "sem_tempos_validos",
        }

    estat = calcular_boxplot_iqr_tempos(tempos)
    tempos_limpos = estat["tempos_limpos"]

    mediana = float(tempos.median())
    tempo_ideal_q1 = float(tempos_limpos.quantile(0.25)) if not tempos_limpos.empty else mediana
    gap = max(0.0, mediana - tempo_ideal_q1)
    gap_perc = round((gap / mediana) * 100, 2) if mediana > 0 else 0.0

    return {
        "mediana_tempo_sec": mediana,
        "mediana_tempo_fmt": format_seconds(mediana),
        "tempo_ideal_q1_sec": tempo_ideal_q1,
        "tempo_ideal_q1_fmt": format_seconds(tempo_ideal_q1),
        "gap_mediana_q1_sec": gap,
        "gap_mediana_q1_fmt": format_seconds(gap),
        "gap_mediana_q1_perc": gap_perc,
        "qtd_outliers_boxplot": int(estat["qtd_outliers"]),
        "qtd_base_limpa_boxplot": int(estat["qtd_base_limpa"]),
        "metodo_ideal_aplicado": str(estat["metodo"]),
    }


# -- Janelas de entrega --------------------------------------------------------

@st.cache_data(show_spinner=False)
def calcular_janelas_entrega(
    processados: pd.DataFrame,
    cobertura: float = 0.80,
    usar_coluna: str = "Chegou_em",
) -> pd.DataFrame:
    if processados is None or processados.empty:
        return pd.DataFrame()

    if usar_coluna not in processados.columns:
        return pd.DataFrame()

    dados = processados.copy()
    dados["Cod_Cliente"] = dados["Cod_Cliente"].astype(str).str.strip()
    dados = dados[dados["Cod_Cliente"].notna() & (dados["Cod_Cliente"] != "")].copy()

    if dados.empty:
        return pd.DataFrame()

    dados["Minutos_Base"] = minutos_desde_meia_noite(dados[usar_coluna])
    resultados = []

    for cliente, grupo in dados.groupby("Cod_Cliente", sort=True):
        minutos = grupo["Minutos_Base"].dropna().tolist()

        if not minutos:
            continue

        janela = calcular_janela_circular_minima(minutos=minutos, cobertura=cobertura)
        media_minutos = float(pd.Series(minutos).mean()) if minutos else None

        resultados.append(
            {
                "Cod_Cliente": str(cliente).strip(),
                "Qtd_Apontamentos": int(len(grupo)),
                "Janela_Inicio_Min": janela["janela_inicio_min"],
                "Janela_Fim_Min": janela["janela_fim_min"],
                "Janela_Inicio": formatar_minutos_hhmm(janela["janela_inicio_min"]),
                "Janela_Fim": formatar_minutos_hhmm(janela["janela_fim_min"]),
                "Janela_Largura_Min": round(float(janela["largura_min"] or 0), 2),
                "Cobertura_Alvo": f"{round(cobertura * 100)}%",
                "Cobertura_Real": f"{round(janela['cobertura_real'] * 100, 1)}%",
                "Cobertura_Real_Valor": float(janela["cobertura_real"]),
                "Cruza_MeiaNoite": "Sim" if janela["cruza_meia_noite"] else "Nao",
                "Periodo_Predominante": classificar_periodo(media_minutos),
                "Comercial": classificar_comercial(media_minutos),
                "Base_Janela": usar_coluna,
            }
        )

    if not resultados:
        return pd.DataFrame()

    return (
        pd.DataFrame(resultados)
        .sort_values(by=["Qtd_Apontamentos", "Cod_Cliente"], ascending=[False, True])
        .reset_index(drop=True)
    )


# -- Dados do cliente ----------------------------------------------------------

def obter_coluna_volume(df: pd.DataFrame) -> str | None:
    candidatos = [
        "Vol_caixas",
        "vol_caixas",
        "Vol caixas",
        "vol caixas",
        "Volume_caixas",
        "Volume de caixas",
        "volume_caixas",
        "vol de caixas",
    ]
    for coluna in candidatos:
        if coluna in df.columns:
            return coluna
    return None


@st.cache_data(show_spinner=False)
def montar_dados_cliente(
    processados: pd.DataFrame,
    medianas: pd.DataFrame,
    cliente: str,
) -> tuple[pd.DataFrame, dict]:
    if processados is None or processados.empty:
        return pd.DataFrame(), {}

    dados_cliente = processados[
        processados["Cod_Cliente"].astype(str).str.strip() == str(cliente).strip()
    ].copy()

    if dados_cliente.empty:
        return pd.DataFrame(), {}

    dados_cliente = dados_cliente.sort_values(by="Chegou_em").reset_index(drop=True)

    coluna_volume = obter_coluna_volume(dados_cliente)
    if coluna_volume is None:
        dados_cliente["Vol_caixas"] = 0.0
        coluna_volume = "Vol_caixas"

    dados_cliente["Vol_caixas_num"], resumo_volume = normalizar_volume_caixas(
        dados_cliente[coluna_volume]
    )

    dados_cliente["Tempo_Sec"] = pd.to_numeric(
        dados_cliente["Tempo_Sec"], errors="coerce"
    ).fillna(0)
    dados_cliente["Data_Entrega_Label"] = dados_cliente["Chegou_em"].dt.strftime("%d/%m/%Y")
    dados_cliente["DataHora_Entrega_Label"] = dados_cliente["Chegou_em"].dt.strftime(
        "%d/%m/%Y %H:%M"
    )
    dados_cliente["Tempo_Formatado"] = dados_cliente["Tempo_Sec"].apply(format_seconds)
    dados_cliente["Vol_caixas_fmt"] = dados_cliente["Vol_caixas_num"].apply(
        lambda x: formatar_numero(x, 2)
    )
    dados_cliente["Ordem_Eixo"] = list(range(1, len(dados_cliente) + 1))
    dados_cliente["Chegou_Min"] = minutos_desde_meia_noite(dados_cliente["Chegou_em"])
    dados_cliente["Finalizada_Min"] = minutos_desde_meia_noite(dados_cliente["Finalizada_em"])
    dados_cliente["Hora_Abertura"] = dados_cliente["Chegou_Min"].apply(formatar_minutos_hhmm)
    dados_cliente["Hora_Finalizacao"] = dados_cliente["Finalizada_Min"].apply(formatar_minutos_hhmm)
    dados_cliente["Dia_Semana_Num"] = pd.to_datetime(
        dados_cliente["Chegou_em"], errors="coerce"
    ).dt.dayofweek
    dados_cliente["Dia_Semana"] = dados_cliente["Dia_Semana_Num"].map(DIAS_SEMANA_MAP).fillna("")

    linha_mediana = medianas[
        medianas["Cod_Cliente"].astype(str).str.strip() == str(cliente).strip()
    ].copy()

    mediana_tempo_sec = 0.0
    mediana_tempo_fmt = "00:00:00"
    tempo_ideal_q1_sec = 0.0
    tempo_ideal_q1_fmt = "00:00:00"
    gap_mediana_q1_sec = 0.0
    gap_mediana_q1_fmt = "00:00:00"
    gap_mediana_q1_perc = 0.0
    qtd_outliers_boxplot = 0
    qtd_base_limpa_boxplot = 0
    metodo_ideal_aplicado = ""

    if not linha_mediana.empty:
        linha = linha_mediana.iloc[0]
        mediana_tempo_sec = float(linha.get("Mediana_Tempo_Sec", 0) or 0)
        mediana_tempo_fmt = str(linha.get("Mediana_Tempo_Formatada", "00:00:00"))
        tempo_ideal_q1_sec = float(
            linha.get("Tempo_Ideal_Q1_Sec", mediana_tempo_sec) or 0
        )
        tempo_ideal_q1_fmt = str(
            linha.get("Tempo_Ideal_Q1_Formatado", format_seconds(tempo_ideal_q1_sec))
        )
        gap_mediana_q1_sec = float(
            linha.get(
                "Diferenca_Mediana_Q1_Sec",
                max(0, mediana_tempo_sec - tempo_ideal_q1_sec),
            )
            or 0
        )
        gap_mediana_q1_fmt = str(
            linha.get("Diferenca_Mediana_Q1_Formatada", format_seconds(gap_mediana_q1_sec))
        )
        gap_mediana_q1_perc = float(linha.get("Diferenca_Mediana_Q1_Percentual", 0) or 0)
        qtd_outliers_boxplot = int(float(linha.get("Qtd_Outliers_Boxplot", 0) or 0))
        qtd_base_limpa_boxplot = int(float(linha.get("Qtd_Base_Limpa_Boxplot", 0) or 0))
        metodo_ideal_aplicado = str(linha.get("Metodo_Ideal_Aplicado", ""))

    dados_cliente["Mediana_Ref"] = mediana_tempo_fmt
    dados_cliente["Tempo_Ideal_Q1_Ref"] = tempo_ideal_q1_fmt

    media_vol_caixas = (
        float(dados_cliente["Vol_caixas_num"].mean()) if not dados_cliente.empty else 0.0
    )

    resumo = {
        "cliente": str(cliente),
        "qtd_entregas": int(len(dados_cliente)),
        "media_vol_caixas": media_vol_caixas,
        "media_vol_caixas_fmt": formatar_numero(media_vol_caixas, 2),
        "mediana_tempo_sec": mediana_tempo_sec,
        "mediana_tempo_fmt": mediana_tempo_fmt,
        "tempo_ideal_q1_sec": tempo_ideal_q1_sec,
        "tempo_ideal_q1_fmt": tempo_ideal_q1_fmt,
        "gap_mediana_q1_sec": gap_mediana_q1_sec,
        "gap_mediana_q1_fmt": gap_mediana_q1_fmt,
        "gap_mediana_q1_perc": gap_mediana_q1_perc,
        "qtd_outliers_boxplot": qtd_outliers_boxplot,
        "qtd_base_limpa_boxplot": qtd_base_limpa_boxplot,
        "metodo_ideal_aplicado": metodo_ideal_aplicado,
        "coluna_volume": coluna_volume,
        "coluna_volume_plot": "Vol_caixas_num",
        "qtd_nulos_origem": resumo_volume["qtd_nulos_origem"],
        "qtd_zeros_reais": resumo_volume["qtd_zeros_reais"],
        "qtd_invalidos_convertidos": resumo_volume["qtd_invalidos_convertidos"],
        "qtd_total_volume": resumo_volume["qtd_total"],
    }

    return dados_cliente, resumo


def filtrar_dados_cliente_por_dia_semana(
    dados_cliente: pd.DataFrame,
    dia_semana: str,
) -> pd.DataFrame:
    if dados_cliente is None or dados_cliente.empty:
        return pd.DataFrame()

    if not dia_semana or dia_semana == "TODOS":
        return dados_cliente.copy().reset_index(drop=True)

    return (
        dados_cliente[
            dados_cliente["Dia_Semana"].astype(str).str.upper()
            == str(dia_semana).strip().upper()
        ]
        .copy()
        .reset_index(drop=True)
    )


def recalcular_resumo_cliente_filtrado(
    dados_cliente: pd.DataFrame,
    resumo_base: dict,
) -> dict:
    resumo = dict(resumo_base)

    if dados_cliente is None or dados_cliente.empty:
        resumo.update(
            {
                "qtd_entregas": 0,
                "media_vol_caixas": 0.0,
                "media_vol_caixas_fmt": formatar_numero(0.0, 2),
                "mediana_tempo_sec": 0.0,
                "mediana_tempo_fmt": "00:00:00",
                "tempo_ideal_q1_sec": 0.0,
                "tempo_ideal_q1_fmt": "00:00:00",
                "gap_mediana_q1_sec": 0.0,
                "gap_mediana_q1_fmt": "00:00:00",
                "gap_mediana_q1_perc": 0.0,
                "qtd_outliers_boxplot": 0,
                "qtd_base_limpa_boxplot": 0,
                "metodo_ideal_aplicado": "sem_tempos_validos",
                "qtd_nulos_origem": 0,
                "qtd_zeros_reais": 0,
                "qtd_invalidos_convertidos": 0,
                "qtd_total_volume": 0,
            }
        )
        return resumo

    coluna_volume = resumo.get("coluna_volume", "Vol_caixas")
    if coluna_volume in dados_cliente.columns:
        _, resumo_volume = normalizar_volume_caixas(dados_cliente[coluna_volume])
    else:
        resumo_volume = {
            "qtd_total": int(len(dados_cliente)),
            "qtd_nulos_origem": 0,
            "qtd_zeros_reais": 0,
            "qtd_invalidos_convertidos": 0,
        }

    tempos_ref = calcular_tempos_referencia_local(dados_cliente["Tempo_Sec"])
    media_vol_caixas = float(
        pd.to_numeric(dados_cliente["Vol_caixas_num"], errors="coerce").fillna(0).mean()
    )

    resumo.update(
        {
            "qtd_entregas": int(len(dados_cliente)),
            "media_vol_caixas": media_vol_caixas,
            "media_vol_caixas_fmt": formatar_numero(media_vol_caixas, 2),
            "mediana_tempo_sec": tempos_ref["mediana_tempo_sec"],
            "mediana_tempo_fmt": tempos_ref["mediana_tempo_fmt"],
            "tempo_ideal_q1_sec": tempos_ref["tempo_ideal_q1_sec"],
            "tempo_ideal_q1_fmt": tempos_ref["tempo_ideal_q1_fmt"],
            "gap_mediana_q1_sec": tempos_ref["gap_mediana_q1_sec"],
            "gap_mediana_q1_fmt": tempos_ref["gap_mediana_q1_fmt"],
            "gap_mediana_q1_perc": tempos_ref["gap_mediana_q1_perc"],
            "qtd_outliers_boxplot": tempos_ref["qtd_outliers_boxplot"],
            "qtd_base_limpa_boxplot": tempos_ref["qtd_base_limpa_boxplot"],
            "metodo_ideal_aplicado": tempos_ref["metodo_ideal_aplicado"],
            "qtd_nulos_origem": resumo_volume["qtd_nulos_origem"],
            "qtd_zeros_reais": resumo_volume["qtd_zeros_reais"],
            "qtd_invalidos_convertidos": resumo_volume["qtd_invalidos_convertidos"],
            "qtd_total_volume": resumo_volume["qtd_total"],
        }
    )

    return resumo


# -- Graficos ------------------------------------------------------------------

def criar_grafico_cliente(
    dados_cliente: pd.DataFrame,
    resumo: dict,
    mostrar_rotulos_tempo: bool = True,
) -> go.Figure:
    dados_plot = dados_cliente.copy()

    x_valores = dados_plot["Ordem_Eixo"].tolist()
    tick_textos = dados_plot["Data_Entrega_Label"].tolist()

    fig = go.Figure()

    fig.add_bar(
        x=x_valores,
        y=dados_plot[resumo["coluna_volume_plot"]].tolist(),
        name="Vol_caixas",
        text=dados_plot["Vol_caixas_fmt"].tolist(),
        textposition="outside",
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Data/Hora: %{customdata[1]}<br>"
            "Vol_caixas: %{y:.2f}<extra></extra>"
        ),
        customdata=dados_plot[["Data_Entrega_Label", "DataHora_Entrega_Label"]].values.tolist(),
        yaxis="y",
    )

    fig.add_scatter(
        x=x_valores,
        y=dados_plot["Tempo_Sec"].tolist(),
        name="Tempo gasto",
        mode="lines+markers+text" if mostrar_rotulos_tempo else "lines+markers",
        text=dados_plot["Tempo_Formatado"].tolist() if mostrar_rotulos_tempo else None,
        textposition="top center",
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Data/Hora: %{customdata[1]}<br>"
            "Tempo: %{customdata[2]}<extra></extra>"
        ),
        customdata=dados_plot[
            ["Data_Entrega_Label", "DataHora_Entrega_Label", "Tempo_Formatado"]
        ].values.tolist(),
        yaxis="y2",
    )

    if x_valores:
        fig.add_scatter(
            x=x_valores,
            y=[resumo["mediana_tempo_sec"]] * len(x_valores),
            name=f"Mediana ({resumo['mediana_tempo_fmt']})",
            mode="lines",
            line=dict(dash="dash"),
            hovertemplate=f"Mediana: {resumo['mediana_tempo_fmt']}<extra></extra>",
            yaxis="y2",
        )

        fig.add_scatter(
            x=x_valores,
            y=[resumo["tempo_ideal_q1_sec"]] * len(x_valores),
            name=f"Tempo ideal Q1 ({resumo['tempo_ideal_q1_fmt']})",
            mode="lines",
            line=dict(dash="dot"),
            hovertemplate=f"Tempo ideal Q1: {resumo['tempo_ideal_q1_fmt']}<extra></extra>",
            yaxis="y2",
        )

    qtd_ticks = len(tick_textos)
    passo = max(1, qtd_ticks // 20) if qtd_ticks else 1
    tick_vals_filtrados = [
        x for i, x in enumerate(x_valores) if i % passo == 0 or i == qtd_ticks - 1
    ]
    tick_text_filtrados = [
        t for i, t in enumerate(tick_textos) if i % passo == 0 or i == qtd_ticks - 1
    ]

    range_inicial = (
        [max(1, len(x_valores) - 14), len(x_valores)]
        if len(x_valores) > 15
        else [1, max(1, len(x_valores))]
    )

    fig.update_layout(
        height=650,
        hovermode="x unified",
        dragmode="pan",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=30, r=30, t=60, b=30),
        xaxis=dict(
            title="Datas das entregas",
            tickmode="array",
            tickvals=tick_vals_filtrados,
            ticktext=tick_text_filtrados,
            rangeslider=dict(visible=True),
            range=range_inicial,
        ),
        yaxis=dict(title="Volume de caixas", rangemode="tozero"),
        yaxis2=dict(
            title="Tempo gasto (segundos)",
            overlaying="y",
            side="right",
            rangemode="tozero",
        ),
    )

    return fig


def criar_grafico_aberturas_cliente(
    dados_cliente: pd.DataFrame,
    janela_cliente: pd.DataFrame,
    base_janela: str,
) -> go.Figure:
    fig = go.Figure()

    if dados_cliente is None or dados_cliente.empty:
        fig.update_layout(height=420, title="Distribuicao das aberturas")
        return fig

    coluna_min = "Chegou_Min" if base_janela == "Chegou_em" else "Finalizada_Min"
    coluna_label = "Hora_Abertura" if base_janela == "Chegou_em" else "Hora_Finalizacao"

    dados_plot = (
        dados_cliente.dropna(subset=[coluna_min])
        .sort_values(coluna_min)
        .reset_index(drop=True)
        .copy()
    )

    if dados_plot.empty:
        fig.update_layout(height=420, title="Distribuicao das aberturas")
        return fig

    dados_plot["Ocorrencia"] = range(1, len(dados_plot) + 1)

    fig.add_scatter(
        x=dados_plot[coluna_min],
        y=dados_plot["Ocorrencia"],
        mode="markers+text",
        name="Aberturas",
        text=dados_plot[coluna_label],
        textposition="top center",
        marker=dict(size=10),
        customdata=dados_plot[["DataHora_Entrega_Label", "Tempo_Formatado"]].values,
        hovertemplate=(
            "Hora: %{text}<br>"
            "Data/Hora: %{customdata[0]}<br>"
            "Tempo: %{customdata[1]}<extra></extra>"
        ),
    )

    if janela_cliente is not None and not janela_cliente.empty:
        linha = janela_cliente.iloc[0]
        inicio = linha["Janela_Inicio_Min"]
        fim = linha["Janela_Fim_Min"]
        cruza = linha["Cruza_MeiaNoite"] == "Sim"

        if not pd.isna(inicio) and not pd.isna(fim):
            if cruza:
                fig.add_vrect(
                    x0=0, x1=fim,
                    fillcolor="green", opacity=0.15, line_width=0,
                    annotation_text="Janela", annotation_position="top left",
                )
                fig.add_vrect(
                    x0=inicio, x1=1440,
                    fillcolor="green", opacity=0.15, line_width=0,
                )
            else:
                fig.add_vrect(
                    x0=inicio, x1=fim,
                    fillcolor="green", opacity=0.15, line_width=0,
                    annotation_text="Janela", annotation_position="top left",
                )

            fig.add_vline(x=inicio, line_dash="dash", line_width=1)
            fig.add_vline(x=fim, line_dash="dash", line_width=1)

    tick_vals = list(range(0, 1441, 60))
    tick_text = [formatar_minutos_hhmm(v) for v in tick_vals]

    fig.update_layout(
        height=430,
        title="Distribuicao das aberturas e faixa da janela",
        showlegend=False,
        margin=dict(l=30, r=30, t=60, b=30),
        xaxis=dict(
            title="Hora do dia",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            range=[0, 1440],
        ),
        yaxis=dict(title="Sequencia de ocorrencias", rangemode="tozero"),
    )

    return fig


# -- Helpers de exibicao -------------------------------------------------------

def exibir_preview_df(
    df: pd.DataFrame,
    titulo: str,
    limite: int = 1000,
    height: int = 320,
) -> None:
    st.subheader(titulo)

    if df is None or df.empty:
        st.info("Sem dados para exibicao.")
        return

    total = len(df)
    if total > limite:
        st.caption(f"Exibindo {limite:,} de {total:,} linhas.".replace(",", "."))
        st.dataframe(
            preparar_dataframe_streamlit(df.head(limite)),
            use_container_width=True,
            height=height,
        )
    else:
        st.dataframe(
            preparar_dataframe_streamlit(df),
            use_container_width=True,
            height=height,
        )


# -- Interface principal -------------------------------------------------------

def main() -> None:
    st.title("Luna")
    st.caption("Analise de tempos operacionais")

    available_files = list_available_unit_files()
    unit_options = available_files if available_files else AVAILABLE_UNITS

    with st.sidebar:
        st.header("Entrada")

        with st.form("form_processamento"):
            unidade = st.selectbox(
                "Selecione a unidade",
                options=unit_options,
                index=0,
            )

            st.caption("Arquivos configurados:")
            if available_files:
                st.write(available_files)
            else:
                st.warning("Nenhum arquivo foi encontrado na pasta data.")

            st.header("Parametros")

            tempo_min_expurgo = st.number_input(
                "Tempo minimo de expurgo (segundos)",
                min_value=1,
                value=300,
                step=1,
            )

            tempo_max_anomalia = st.number_input(
                "Tempo maximo para anomalia (segundos)",
                min_value=1,
                value=21600,
                step=1,
            )

            eventos_previos = st.number_input(
                "Eventos previos",
                min_value=1,
                value=10,
                step=1,
            )

            minimo_apontamentos = st.number_input(
                "Minimo de apontamentos por cliente",
                min_value=1,
                value=4,
                step=1,
            )

            tempo_padrao_poucos_apontamentos = st.number_input(
                "Tempo padrao para poucos apontamentos (segundos)",
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

            st.header("Janela de entrega")

            exibir_janelas_entrega = st.checkbox(
                "Exibir janelas de entrega",
                value=True,
            )

            cobertura_janela = st.slider(
                "Cobertura da janela",
                min_value=50,
                max_value=95,
                value=80,
                step=5,
            )

            base_janela = st.selectbox(
                "Base para calculo da janela",
                options=["Chegou_em", "Finalizada_em"],
                index=0,
            )

            processar = st.form_submit_button("Processar base", use_container_width=True)

    assinatura_atual = {
        "unidade": unidade,
        "tempo_min_expurgo": tempo_min_expurgo,
        "tempo_max_anomalia": tempo_max_anomalia,
        "eventos_previos": eventos_previos,
        "minimo_apontamentos": minimo_apontamentos,
        "tempo_padrao_poucos_apontamentos": tempo_padrao_poucos_apontamentos,
        "ajuste_percentual": ajuste_percentual,
        "exibir_janelas_entrega": exibir_janelas_entrega,
        "cobertura_janela": cobertura_janela,
        "base_janela": base_janela,
    }

    if processar:
        st.cache_data.clear()
        st.session_state["assinatura_processamento"] = assinatura_atual
        st.session_state["ultima_unidade_processada"] = unidade
        st.session_state.pop("excel_bytes", None)
        st.session_state.pop("zip_bytes", None)

    tab_base, tab_validacao, tab_processamento, tab_cliente, tab_resultados, tab_exportacao = (
        st.tabs(["Base", "Validacao", "Processamento", "Painel do Cliente", "Resultados", "Exportacao"])
    )

    if "assinatura_processamento" not in st.session_state:
        with tab_base:
            st.info("Selecione a unidade, ajuste os parametros e clique em 'Processar base'.")
        return

    if st.session_state.get("assinatura_processamento") != assinatura_atual:
        st.sidebar.warning("Ha alteracoes de unidade/parametros ainda nao processadas.")

    try:
        with st.spinner("Processando base..."):
            dados_processados = processar_base(
                unidade=st.session_state["assinatura_processamento"]["unidade"],
                tempo_min_expurgo=st.session_state["assinatura_processamento"]["tempo_min_expurgo"],
                tempo_max_anomalia=st.session_state["assinatura_processamento"]["tempo_max_anomalia"],
                eventos_previos=st.session_state["assinatura_processamento"]["eventos_previos"],
                minimo_apontamentos=st.session_state["assinatura_processamento"][
                    "minimo_apontamentos"
                ],
                tempo_padrao_poucos_apontamentos=st.session_state["assinatura_processamento"][
                    "tempo_padrao_poucos_apontamentos"
                ],
                ajuste_percentual=st.session_state["assinatura_processamento"]["ajuste_percentual"],
            )
    except Exception as exc:
        st.error(str(exc))
        return

    base_bruta = dados_processados["base_bruta"]
    base_padronizada = dados_processados["base_padronizada"]
    schema_report = dados_processados["schema_report"]
    relatorio_validacao = dados_processados["relatorio_validacao"]
    processados = dados_processados["processados"]
    inconsistencias = dados_processados["inconsistencias"]
    expurgados = dados_processados["expurgados"]
    anomalias = dados_processados["anomalias"]
    medianas = dados_processados["medianas"]
    kpis = dados_processados["kpis"]

    base_detalhada = montar_base_detalhada(processados=processados, medianas=medianas)

    janelas_entrega = pd.DataFrame()
    if st.session_state["assinatura_processamento"].get("exibir_janelas_entrega", False):
        janelas_entrega = calcular_janelas_entrega(
            processados=processados,
            cobertura=st.session_state["assinatura_processamento"]["cobertura_janela"] / 100,
            usar_coluna=st.session_state["assinatura_processamento"]["base_janela"],
        )

    with tab_base:
        st.subheader("Visao da base")

        col1, col2, col3 = st.columns(3)
        col1.metric("Linhas brutas", len(base_bruta))
        col2.metric("Colunas encontradas", relatorio_validacao["total_columns"])
        col3.metric("Estrutura valida", "Sim" if relatorio_validacao["is_valid"] else "Nao")

        exibir_preview_df(base_padronizada, "Pre-visualizacao da base", limite=100, height=450)

    with tab_validacao:
        st.subheader("Validacao estrutural")

        if relatorio_validacao["is_valid"]:
            st.success("Estrutura valida para processamento.")
        else:
            st.error("Estrutura invalida.")
            st.write("Colunas obrigatorias ausentes:")
            st.write(relatorio_validacao["required_missing"])

            sugestoes = suggest_missing_columns(schema_report)
            if sugestoes:
                st.markdown("### Sugestoes")
                for sugestao in sugestoes:
                    st.write(f"- {sugestao}")

            st.stop()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Colunas obrigatorias encontradas")
            st.dataframe(
                preparar_dataframe_streamlit(
                    pd.DataFrame({"Colunas": relatorio_validacao["required_found"]})
                ),
                use_container_width=True,
                height=180,
            )

        with col2:
            st.markdown("### Colunas extras nao reconhecidas")
            st.dataframe(
                preparar_dataframe_streamlit(
                    pd.DataFrame({"Colunas": relatorio_validacao["unknown_columns"]})
                ),
                use_container_width=True,
                height=180,
            )

        if relatorio_validacao["duplicate_standardized_columns"]:
            st.warning("Foram detectadas colunas duplicadas apos a padronizacao:")
            st.write(relatorio_validacao["duplicate_standardized_columns"])

        exibir_preview_df(
            pd.DataFrame(relatorio_validacao["mapping_preview"]),
            "Mapeamento aplicado",
            limite=1000,
            height=320,
        )

        st.subheader("Schema oficial")
        st.dataframe(
            preparar_dataframe_streamlit(get_schema_dataframe()),
            use_container_width=True,
            height=260,
        )

        exibir_preview_df(get_aliases_dataframe(), "Aliases reconhecidos", limite=1000, height=320)

    with tab_processamento:
        st.subheader("KPIs do processamento")

        col1, col2, col3, col4 = st.columns(4)
        col5, col6, col7, col8 = st.columns(4)

        col1.metric("Linhas validas", kpis.get("linhas_validas", 0))
        col2.metric("Inconsistencias", kpis.get("inconsistencias", 0))
        col3.metric("Expurgados", kpis.get("expurgados", 0))
        col4.metric("Anomalias", kpis.get("anomalias", 0))
        col5.metric("Clientes unicos", kpis.get("clientes_unicos", 0))
        col6.metric("Mediana global", kpis.get("mediana_global_fmt", "00:00:00"))
        col7.metric("Tempo ideal global Q1", kpis.get("tempo_ideal_q1_global_fmt", "00:00:00"))
        col8.metric("Outliers Boxplot", kpis.get("outliers_boxplot", 0))

        st.subheader("Resumo")
        st.write(
            {
                "unidade": st.session_state.get("ultima_unidade_processada", unidade),
                "linhas_brutas": kpis.get("linhas_brutas", 0),
                "linhas_validas": kpis.get("linhas_validas", 0),
                "inconsistencias": kpis.get("inconsistencias", 0),
                "expurgados": kpis.get("expurgados", 0),
                "anomalias": kpis.get("anomalias", 0),
                "clientes_unicos": kpis.get("clientes_unicos", 0),
                "mediana_global": kpis.get("mediana_global_fmt", "00:00:00"),
                "tempo_ideal_global_q1": kpis.get("tempo_ideal_q1_global_fmt", "00:00:00"),
                "gap_global_mediana_q1": kpis.get("gap_mediana_q1_global_fmt", "00:00:00"),
                "outliers_boxplot": kpis.get("outliers_boxplot", 0),
            }
        )

        if st.session_state["assinatura_processamento"].get("exibir_janelas_entrega", False):
            st.subheader("Parametros das janelas")
            st.write(
                {
                    "exibir_janelas_entrega": "Sim",
                    "cobertura_janela": f"{st.session_state['assinatura_processamento']['cobertura_janela']}%",
                    "base_janela": st.session_state["assinatura_processamento"]["base_janela"],
                    "qtd_janelas_geradas": len(janelas_entrega),
                }
            )

    with tab_cliente:
        st.subheader("Evolucao de tempos por cliente")

        if processados is None or processados.empty:
            st.info("Nao ha dados processados para exibir o painel do cliente.")
        else:
            clientes_disponiveis = sorted(
                processados["Cod_Cliente"].astype(str).str.strip().dropna().unique().tolist()
            )

            if not clientes_disponiveis:
                st.info("Nao ha clientes disponiveis apos o processamento.")
            else:
                cliente_default = 0
                if "cliente_selecionado" in st.session_state:
                    cliente_atual = str(st.session_state["cliente_selecionado"]).strip()
                    if cliente_atual in clientes_disponiveis:
                        cliente_default = clientes_disponiveis.index(cliente_atual)

                col_filtro_1, col_filtro_2 = st.columns([3, 1])

                with col_filtro_1:
                    cliente = st.selectbox(
                        "Selecione o cliente",
                        options=clientes_disponiveis,
                        index=cliente_default,
                        key="cliente_selecionado",
                    )

                with col_filtro_2:
                    mostrar_rotulos_tempo = st.checkbox(
                        "Mostrar rotulos",
                        value=True,
                        key="mostrar_rotulos_tempo_cliente",
                    )

                filtro_dia_semana = st.segmented_control(
                    "Dia da semana",
                    options=OPCOES_DIA_SEMANA,
                    selection_mode="single",
                    default=st.session_state.get("filtro_dia_semana_cliente", "TODOS"),
                    key="filtro_dia_semana_cliente",
                    width="stretch",
                )

                if filtro_dia_semana is None:
                    filtro_dia_semana = "TODOS"

                dados_cliente, resumo_cliente = montar_dados_cliente(
                    processados=processados,
                    medianas=medianas,
                    cliente=cliente,
                )

                dados_cliente = filtrar_dados_cliente_por_dia_semana(
                    dados_cliente=dados_cliente,
                    dia_semana=filtro_dia_semana,
                )

                resumo_cliente = recalcular_resumo_cliente_filtrado(
                    dados_cliente=dados_cliente,
                    resumo_base=resumo_cliente,
                )

                janela_cliente = pd.DataFrame()
                if (
                    st.session_state["assinatura_processamento"].get(
                        "exibir_janelas_entrega", False
                    )
                    and not dados_cliente.empty
                ):
                    janela_cliente = calcular_janelas_entrega(
                        processados=dados_cliente,
                        cobertura=st.session_state["assinatura_processamento"]["cobertura_janela"]
                        / 100,
                        usar_coluna=st.session_state["assinatura_processamento"]["base_janela"],
                    )
                    if not janela_cliente.empty:
                        janela_cliente = janela_cliente[
                            janela_cliente["Cod_Cliente"].astype(str).str.strip()
                            == str(cliente).strip()
                        ].copy()

                if dados_cliente.empty:
                    st.warning(
                        f"Nao ha dados validos para o cliente selecionado com o filtro '{filtro_dia_semana}'."
                    )
                else:
                    st.caption(f"Filtro de dia aplicado: {filtro_dia_semana}")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Qtd. entregas", resumo_cliente["qtd_entregas"])
                    col2.metric("Mediana do tempo", resumo_cliente["mediana_tempo_fmt"])
                    col3.metric("Tempo ideal Q1", resumo_cliente["tempo_ideal_q1_fmt"])
                    col4.metric(
                        "Gap Mediana x Q1",
                        resumo_cliente["gap_mediana_q1_fmt"],
                        f"{resumo_cliente['gap_mediana_q1_perc']}%",
                    )

                    col5, col6, col7 = st.columns(3)
                    col5.metric("Media de Vol_caixas", resumo_cliente["media_vol_caixas_fmt"])
                    col6.metric("Base limpa Boxplot", resumo_cliente["qtd_base_limpa_boxplot"])
                    col7.metric("Outliers Boxplot", resumo_cliente["qtd_outliers_boxplot"])

                    if resumo_cliente["qtd_invalidos_convertidos"] > 0:
                        st.warning(
                            f"Foram encontrados {resumo_cliente['qtd_invalidos_convertidos']} "
                            "registros de volume que nao puderam ser convertidos corretamente "
                            "e foram exibidos como 0 no grafico."
                        )

                    st.caption(
                        "Validacao de Vol_caixas — "
                        f"Total: {resumo_cliente['qtd_total_volume']} | "
                        f"Zeros reais: {resumo_cliente['qtd_zeros_reais']} | "
                        f"Nulos na origem: {resumo_cliente['qtd_nulos_origem']} | "
                        f"Invalidos convertidos: {resumo_cliente['qtd_invalidos_convertidos']}"
                    )

                    st.caption(
                        "Referencias de tempo — "
                        f"Mediana: {resumo_cliente['mediana_tempo_fmt']} | "
                        f"Tempo ideal Q1: {resumo_cliente['tempo_ideal_q1_fmt']} | "
                        f"Gap: {resumo_cliente['gap_mediana_q1_fmt']} "
                        f"({resumo_cliente['gap_mediana_q1_perc']}%) | "
                        f"Metodo ideal: {resumo_cliente['metodo_ideal_aplicado']}"
                    )

                    grafico = criar_grafico_cliente(
                        dados_cliente=dados_cliente,
                        resumo=resumo_cliente,
                        mostrar_rotulos_tempo=mostrar_rotulos_tempo,
                    )
                    st.plotly_chart(grafico, use_container_width=True)

                    if st.session_state["assinatura_processamento"].get(
                        "exibir_janelas_entrega", False
                    ):
                        st.subheader("Janela de entrega do cliente")
                        base_janela_ativa = st.session_state["assinatura_processamento"][
                            "base_janela"
                        ]

                        if janela_cliente.empty:
                            st.warning(
                                "Nao foi possivel calcular a janela para este cliente com o filtro atual. "
                                "O grafico abaixo sera exibido somente com as aberturas/finalizacoes."
                            )
                            grafico_aberturas = criar_grafico_aberturas_cliente(
                                dados_cliente=dados_cliente,
                                janela_cliente=pd.DataFrame(),
                                base_janela=base_janela_ativa,
                            )
                            st.plotly_chart(grafico_aberturas, use_container_width=True)
                        else:
                            linha_janela = janela_cliente.iloc[0]

                            cj1, cj2, cj3, cj4, cj5, cj6 = st.columns(6)
                            cj1.metric("Inicio da janela", linha_janela["Janela_Inicio"])
                            cj2.metric("Fim da janela", linha_janela["Janela_Fim"])
                            cj3.metric(
                                "Largura",
                                f"{int(round(float(linha_janela['Janela_Largura_Min'])))} min",
                            )
                            cj4.metric("Cobertura real", linha_janela["Cobertura_Real"])
                            cj5.metric("Cruza meia-noite", linha_janela["Cruza_MeiaNoite"])
                            cj6.metric("Comercial", linha_janela["Comercial"])

                            st.caption(
                                f"Base utilizada: {linha_janela['Base_Janela']} | "
                                f"Periodo predominante: {linha_janela['Periodo_Predominante']} | "
                                f"Comercial: {linha_janela['Comercial']}"
                            )

                            grafico_aberturas = criar_grafico_aberturas_cliente(
                                dados_cliente=dados_cliente,
                                janela_cliente=janela_cliente,
                                base_janela=base_janela_ativa,
                            )
                            st.plotly_chart(grafico_aberturas, use_container_width=True)

                    if "Mediana_Ref" not in dados_cliente.columns:
                        dados_cliente["Mediana_Ref"] = resumo_cliente["mediana_tempo_fmt"]
                    if "Tempo_Ideal_Q1_Ref" not in dados_cliente.columns:
                        dados_cliente["Tempo_Ideal_Q1_Ref"] = resumo_cliente["tempo_ideal_q1_fmt"]

                    colunas_tabela = [
                        "DataHora_Entrega_Label",
                        "Dia_Semana",
                        "Hora_Abertura",
                        "Hora_Finalizacao",
                        "Tempo_Formatado",
                        "Mediana_Ref",
                        "Tempo_Ideal_Q1_Ref",
                        "Vol_caixas_num",
                    ]
                    if "tour_display_id" in dados_cliente.columns:
                        colunas_tabela.append("tour_display_id")

                    exibicao_cliente = dados_cliente[colunas_tabela].copy()

                    rename_map = {
                        "DataHora_Entrega_Label": "Data da entrega",
                        "Dia_Semana": "Dia da semana",
                        "Hora_Abertura": "Abertura",
                        "Hora_Finalizacao": "Finalizacao",
                        "Tempo_Formatado": "Tempo gasto",
                        "Mediana_Ref": "Mediana",
                        "Tempo_Ideal_Q1_Ref": "Tempo ideal Q1",
                        "Vol_caixas_num": "Vol_caixas",
                    }
                    if "tour_display_id" in exibicao_cliente.columns:
                        rename_map["tour_display_id"] = "Tour"

                    exibicao_cliente = exibicao_cliente.rename(columns=rename_map)

                    linha_mediana_row = {
                        "Data da entrega": "Resumo - Mediana",
                        "Dia da semana": filtro_dia_semana,
                        "Abertura": "-",
                        "Finalizacao": "-",
                        "Tempo gasto": resumo_cliente["mediana_tempo_fmt"],
                        "Mediana": resumo_cliente["mediana_tempo_fmt"],
                        "Tempo ideal Q1": resumo_cliente["tempo_ideal_q1_fmt"],
                        "Vol_caixas": resumo_cliente["media_vol_caixas"],
                    }
                    linha_q1_row = {
                        "Data da entrega": "Resumo - Tempo ideal Q1",
                        "Dia da semana": filtro_dia_semana,
                        "Abertura": "-",
                        "Finalizacao": "-",
                        "Tempo gasto": resumo_cliente["tempo_ideal_q1_fmt"],
                        "Mediana": resumo_cliente["mediana_tempo_fmt"],
                        "Tempo ideal Q1": resumo_cliente["tempo_ideal_q1_fmt"],
                        "Vol_caixas": resumo_cliente["media_vol_caixas"],
                    }
                    if "Tour" in exibicao_cliente.columns:
                        linha_mediana_row["Tour"] = "-"
                        linha_q1_row["Tour"] = "-"

                    exibicao_cliente = pd.concat(
                        [exibicao_cliente, pd.DataFrame([linha_mediana_row, linha_q1_row])],
                        ignore_index=True,
                    )

                    st.dataframe(
                        preparar_dataframe_streamlit(exibicao_cliente),
                        use_container_width=True,
                        height=380,
                    )

    with tab_resultados:
        exibir_preview_df(medianas, "Medianas por cliente", limite=1000, height=320)

        exibir_preview_df(
            base_detalhada,
            "Base detalhada (entregas com referencias por cliente)",
            limite=1000,
            height=320,
        )

        if st.session_state["assinatura_processamento"].get("exibir_janelas_entrega", False):
            colunas_janela_exibir = [
                "Cod_Cliente",
                "Qtd_Apontamentos",
                "Janela_Inicio",
                "Janela_Fim",
                "Janela_Largura_Min",
                "Cobertura_Alvo",
                "Cobertura_Real",
                "Cruza_MeiaNoite",
                "Periodo_Predominante",
                "Comercial",
                "Base_Janela",
            ]
            exibir_preview_df(
                janelas_entrega[colunas_janela_exibir]
                if not janelas_entrega.empty
                else janelas_entrega,
                "Janelas de entrega",
                limite=1000,
                height=320,
            )

        exibir_preview_df(inconsistencias, "Inconsistencias", limite=1000, height=240)
        exibir_preview_df(expurgados, "Expurgados", limite=1000, height=240)
        exibir_preview_df(anomalias, "Anomalias", limite=1000, height=240)

    with tab_exportacao:
        st.subheader("Exportacao")
        st.caption("Download rapido: ZIP com CSVs")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Preparar ZIP rapido (CSV)", use_container_width=True):
                with st.spinner("Gerando ZIP com CSVs..."):
                    st.session_state["zip_bytes"] = exportar_zip_csv(
                        base_bruta=base_padronizada,
                        base_validos=processados,
                        inconsistencias=inconsistencias,
                        expurgados=expurgados,
                        anomalias=anomalias,
                        medianas=medianas,
                        janelas_atendimento=janelas_entrega,
                        base_detalhada=base_detalhada,
                    ).getvalue()

            if "zip_bytes" in st.session_state:
                st.download_button(
                    label="Baixar ZIP com CSVs",
                    data=st.session_state["zip_bytes"],
                    file_name=f"{st.session_state.get('ultima_unidade_processada', unidade)}_luna_resultado.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

        with col2:
            if st.button("Preparar Excel consolidado", use_container_width=True):
                with st.spinner("Gerando Excel..."):
                    st.session_state["excel_bytes"] = exportar_excel(
                        base_bruta=base_padronizada,
                        base_validos=processados,
                        inconsistencias=inconsistencias,
                        expurgados=expurgados,
                        anomalias=anomalias,
                        medianas=medianas,
                        janelas_atendimento=janelas_entrega,
                        base_detalhada=base_detalhada,
                    ).getvalue()

            if "excel_bytes" in st.session_state:
                st.download_button(
                    label="Baixar Excel consolidado",
                    data=st.session_state["excel_bytes"],
                    file_name=f"{st.session_state.get('ultima_unidade_processada', unidade)}_luna_resultado.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
