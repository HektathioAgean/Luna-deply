import math
import os
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
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

os.environ["OMP_NUM_THREADS"] = "1"

st.set_page_config(
    page_title=APP_TITLE,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)


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
        raise ValueError("A base está vazia ou não pôde ser carregada.")

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


def formatar_numero(value: float | int | None, casas: int = 2) -> str:
    if value is None or pd.isna(value):
        return "0"
    return f"{float(value):,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


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


def normalizar_numero_texto(value) -> float | None:
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    texto = str(value).strip()

    if texto == "" or texto.lower() in {"nan", "none", "<na>"}:
        return None

    texto = texto.replace("\xa0", "")
    texto = texto.replace(" ", "")
    texto = re.sub(r"[^0-9,.\-]", "", texto)

    if texto in {"", "-", ".", ","}:
        return None

    if "," in texto and "." in texto:
        if texto.rfind(",") > texto.rfind("."):
            texto = texto.replace(".", "")
            texto = texto.replace(",", ".")
        else:
            texto = texto.replace(",", "")
    elif "," in texto:
        if texto.count(",") > 1:
            ultima = texto.rfind(",")
            texto = texto[:ultima].replace(",", "") + "." + texto[ultima + 1 :]
        else:
            texto = texto.replace(",", ".")
    elif "." in texto:
        if texto.count(".") > 1:
            ultima = texto.rfind(".")
            texto = texto[:ultima].replace(".", "") + "." + texto[ultima + 1 :]

    try:
        return float(texto)
    except Exception:
        return None


def normalizar_volume_caixas(serie: pd.Series) -> tuple[pd.Series, dict]:
    serie_original = serie.copy()

    if pd.api.types.is_numeric_dtype(serie_original):
        serie_numerica = pd.to_numeric(serie_original, errors="coerce")
    else:
        serie_numerica = serie_original.apply(normalizar_numero_texto)
        serie_numerica = pd.to_numeric(serie_numerica, errors="coerce")

    mask_nulo_original = serie_original.isna()
    mask_zero_real = serie_numerica.fillna(0).eq(0) & ~mask_nulo_original
    mask_invalido_convertido = serie_numerica.isna() & ~mask_nulo_original

    resumo_validacao = {
        "qtd_total": int(len(serie_original)),
        "qtd_nulos_origem": int(mask_nulo_original.sum()),
        "qtd_zeros_reais": int(mask_zero_real.sum()),
        "qtd_invalidos_convertidos": int(mask_invalido_convertido.sum()),
    }

    serie_final = serie_numerica.fillna(0.0)

    return serie_final, resumo_validacao


def value_is_null(valor) -> bool:
    return valor is None or pd.isna(valor)


def minutos_desde_meia_noite(serie: pd.Series) -> pd.Series:
    serie_dt = pd.to_datetime(serie, errors="coerce")
    return (serie_dt.dt.hour * 60) + (serie_dt.dt.minute) + (serie_dt.dt.second / 60.0)


def formatar_minutos_hhmm(valor: float | int | None) -> str:
    if value_is_null(valor):
        return ""

    total_min = int(round(float(valor))) % (24 * 60)
    horas = total_min // 60
    minutos = total_min % 60
    return f"{horas:02d}:{minutos:02d}"


def classificar_periodo(hora_media: float | int | None) -> str:
    if value_is_null(hora_media):
        return ""

    hora = (float(hora_media) % 1440) / 60

    if 7 <= hora < 15:
        return "Diurno"
    if 15 <= hora < 21:
        return "Vespertino"
    return "Noturno"


def calcular_janela_circular_minima(minutos: list[float], cobertura: float = 0.80) -> dict:
    if not minutos:
        return {
            "janela_inicio_min": None,
            "janela_fim_min": None,
            "largura_min": None,
            "cobertura_real": 0.0,
            "cruza_meia_noite": False,
        }

    minutos = sorted([float(x) % 1440 for x in minutos if not pd.isna(x)])
    n = len(minutos)

    if n == 0:
        return {
            "janela_inicio_min": None,
            "janela_fim_min": None,
            "largura_min": None,
            "cobertura_real": 0.0,
            "cruza_meia_noite": False,
        }

    if n == 1:
        unico = minutos[0]
        return {
            "janela_inicio_min": unico,
            "janela_fim_min": unico,
            "largura_min": 0.0,
            "cobertura_real": 1.0,
            "cruza_meia_noite": False,
        }

    qtd_cobertura = max(1, int(math.ceil(n * cobertura)))
    minutos_ext = minutos + [m + 1440 for m in minutos]

    melhor_inicio = None
    melhor_fim = None
    menor_largura = None

    for i in range(n):
        j = i + qtd_cobertura - 1
        if j >= len(minutos_ext):
            break

        inicio = minutos_ext[i]
        fim = minutos_ext[j]
        largura = fim - inicio

        if menor_largura is None or largura < menor_largura:
            menor_largura = largura
            melhor_inicio = inicio % 1440
            melhor_fim = fim % 1440

    cruza = False
    if melhor_inicio is not None and melhor_fim is not None:
        cruza = melhor_fim < melhor_inicio

    return {
        "janela_inicio_min": melhor_inicio,
        "janela_fim_min": melhor_fim,
        "largura_min": menor_largura,
        "cobertura_real": qtd_cobertura / n if n > 0 else 0.0,
        "cruza_meia_noite": cruza,
    }


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
    dados = dados[dados["Cod_Cliente"].notna()].copy()
    dados = dados[dados["Cod_Cliente"] != ""].copy()

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
                "Cruza_MeiaNoite": "Sim" if janela["cruza_meia_noite"] else "Não",
                "Periodo_Predominante": classificar_periodo(media_minutos),
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


@st.cache_data(show_spinner=False)
def montar_dados_cliente(
    processados: pd.DataFrame,
    medianas: pd.DataFrame,
    cliente: str,
) -> tuple[pd.DataFrame, dict]:
    if processados is None or processados.empty:
        return pd.DataFrame(), {}

    dados_cliente = processados[processados["Cod_Cliente"].astype(str).str.strip() == str(cliente).strip()].copy()

    if dados_cliente.empty:
        return pd.DataFrame(), {}

    dados_cliente = dados_cliente.sort_values(by="Chegou_em").reset_index(drop=True)

    coluna_volume = obter_coluna_volume(dados_cliente)
    if coluna_volume is None:
        dados_cliente["Vol_caixas"] = 0.0
        coluna_volume = "Vol_caixas"

    dados_cliente["Vol_caixas_num"], resumo_volume = normalizar_volume_caixas(dados_cliente[coluna_volume])

    dados_cliente["Tempo_Sec"] = pd.to_numeric(dados_cliente["Tempo_Sec"], errors="coerce").fillna(0)
    dados_cliente["Data_Entrega_Label"] = dados_cliente["Chegou_em"].dt.strftime("%d/%m/%Y")
    dados_cliente["DataHora_Entrega_Label"] = dados_cliente["Chegou_em"].dt.strftime("%d/%m/%Y %H:%M")
    dados_cliente["Tempo_Formatado"] = dados_cliente["Tempo_Sec"].apply(format_seconds)
    dados_cliente["Vol_caixas_fmt"] = dados_cliente["Vol_caixas_num"].apply(lambda x: formatar_numero(x, 2))
    dados_cliente["Ordem_Eixo"] = list(range(1, len(dados_cliente) + 1))
    dados_cliente["Chegou_Min"] = minutos_desde_meia_noite(dados_cliente["Chegou_em"])
    dados_cliente["Finalizada_Min"] = minutos_desde_meia_noite(dados_cliente["Finalizada_em"])
    dados_cliente["Hora_Abertura"] = dados_cliente["Chegou_Min"].apply(formatar_minutos_hhmm)
    dados_cliente["Hora_Finalizacao"] = dados_cliente["Finalizada_Min"].apply(formatar_minutos_hhmm)

    linha_mediana = medianas[medianas["Cod_Cliente"].astype(str).str.strip() == str(cliente).strip()].copy()

    mediana_tempo_sec = 0.0
    mediana_tempo_fmt = "00:00:00"
    if not linha_mediana.empty:
        mediana_tempo_sec = float(linha_mediana.iloc[0]["Mediana_Tempo_Sec"])
        mediana_tempo_fmt = str(linha_mediana.iloc[0]["Mediana_Tempo_Formatada"])

    media_vol_caixas = float(dados_cliente["Vol_caixas_num"].mean()) if not dados_cliente.empty else 0.0

    resumo = {
        "cliente": str(cliente),
        "qtd_entregas": int(len(dados_cliente)),
        "media_vol_caixas": media_vol_caixas,
        "media_vol_caixas_fmt": formatar_numero(media_vol_caixas, 2),
        "mediana_tempo_sec": mediana_tempo_sec,
        "mediana_tempo_fmt": mediana_tempo_fmt,
        "coluna_volume": coluna_volume,
        "coluna_volume_plot": "Vol_caixas_num",
        "qtd_nulos_origem": resumo_volume["qtd_nulos_origem"],
        "qtd_zeros_reais": resumo_volume["qtd_zeros_reais"],
        "qtd_invalidos_convertidos": resumo_volume["qtd_invalidos_convertidos"],
        "qtd_total_volume": resumo_volume["qtd_total"],
    }

    return dados_cliente, resumo


def criar_grafico_cliente(
    dados_cliente: pd.DataFrame,
    resumo: dict,
    mostrar_rotulos_tempo: bool = True,
) -> go.Figure:
    dados_plot = dados_cliente.copy()

    resumo_x = len(dados_plot) + 1
    x_valores = dados_plot["Ordem_Eixo"].tolist() + [resumo_x]
    tick_textos = dados_plot["Data_Entrega_Label"].tolist() + ["Média/Mediana"]

    texto_tempo = dados_plot["Tempo_Formatado"].tolist() + [resumo["mediana_tempo_fmt"]]
    texto_volume = dados_plot["Vol_caixas_fmt"].tolist() + [resumo["media_vol_caixas_fmt"]]

    fig = go.Figure()

    fig.add_bar(
        x=x_valores,
        y=dados_plot[resumo["coluna_volume_plot"]].tolist() + [resumo["media_vol_caixas"]],
        name="Vol_caixas",
        text=texto_volume,
        textposition="outside",
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Data/Hora: %{customdata[1]}<br>"
            "Vol_caixas: %{y:.2f}<extra></extra>"
        ),
        customdata=(
            dados_plot[["Data_Entrega_Label", "DataHora_Entrega_Label"]].values.tolist()
            + [["Média/Mediana", "Resumo do cliente"]]
        ),
        yaxis="y",
    )

    fig.add_scatter(
        x=x_valores,
        y=dados_plot["Tempo_Sec"].tolist() + [resumo["mediana_tempo_sec"]],
        name="Tempo gasto",
        mode="lines+markers+text" if mostrar_rotulos_tempo else "lines+markers",
        text=texto_tempo if mostrar_rotulos_tempo else None,
        textposition="top center",
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Data/Hora: %{customdata[1]}<br>"
            "Tempo: %{customdata[2]}<extra></extra>"
        ),
        customdata=(
            dados_plot[["Data_Entrega_Label", "DataHora_Entrega_Label", "Tempo_Formatado"]].values.tolist()
            + [["Média/Mediana", "Resumo do cliente", resumo["mediana_tempo_fmt"]]]
        ),
        yaxis="y2",
    )

    qtd_ticks = len(tick_textos)
    passo = max(1, qtd_ticks // 20)
    tick_vals_filtrados = [x for i, x in enumerate(x_valores) if i % passo == 0 or i == qtd_ticks - 1]
    tick_text_filtrados = [t for i, t in enumerate(tick_textos) if i % passo == 0 or i == qtd_ticks - 1]

    range_inicial = [max(1, resumo_x - 15), resumo_x] if resumo_x > 15 else [1, resumo_x]

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
        yaxis=dict(
            title="Volume de caixas",
            rangemode="tozero",
        ),
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
        fig.update_layout(height=420, title="Distribuição das aberturas")
        return fig

    coluna_min = "Chegou_Min" if base_janela == "Chegou_em" else "Finalizada_Min"
    coluna_label = "Hora_Abertura" if base_janela == "Chegou_em" else "Hora_Finalizacao"

    dados_plot = dados_cliente.copy()
    dados_plot = dados_plot.dropna(subset=[coluna_min]).sort_values(coluna_min).reset_index(drop=True)

    if dados_plot.empty:
        fig.update_layout(height=420, title="Distribuição das aberturas")
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
                    x0=0,
                    x1=fim,
                    fillcolor="green",
                    opacity=0.15,
                    line_width=0,
                    annotation_text="Janela",
                    annotation_position="top left",
                )
                fig.add_vrect(
                    x0=inicio,
                    x1=1440,
                    fillcolor="green",
                    opacity=0.15,
                    line_width=0,
                )
            else:
                fig.add_vrect(
                    x0=inicio,
                    x1=fim,
                    fillcolor="green",
                    opacity=0.15,
                    line_width=0,
                    annotation_text="Janela",
                    annotation_position="top left",
                )

            fig.add_vline(x=inicio, line_dash="dash", line_width=1)
            fig.add_vline(x=fim, line_dash="dash", line_width=1)

    tick_vals = list(range(0, 1441, 60))
    tick_text = [formatar_minutos_hhmm(v) for v in tick_vals]

    fig.update_layout(
        height=430,
        title="Distribuição das aberturas e faixa da janela",
        showlegend=False,
        margin=dict(l=30, r=30, t=60, b=30),
        xaxis=dict(
            title="Hora do dia",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            range=[0, 1440],
        ),
        yaxis=dict(
            title="Sequência de ocorrências",
            rangemode="tozero",
        ),
    )

    return fig


def exibir_preview_df(df: pd.DataFrame, titulo: str, limite: int = 1000, height: int = 320) -> None:
    st.subheader(titulo)

    if df is None or df.empty:
        st.info("Sem dados para exibição.")
        return

    total = len(df)
    if total > limite:
        st.caption(f"Exibindo {limite:,} de {total:,} linhas.".replace(",", "."))
        st.dataframe(df.head(limite), use_container_width=True, height=height)
    else:
        st.dataframe(df, use_container_width=True, height=height)


def preparar_zip_download(
    base_padronizada: pd.DataFrame,
    processados: pd.DataFrame,
    inconsistencias: pd.DataFrame,
    expurgados: pd.DataFrame,
    anomalias: pd.DataFrame,
    medianas: pd.DataFrame,
) -> bytes:
    return exportar_zip_csv(
        base_bruta=base_padronizada,
        base_validos=processados,
        inconsistencias=inconsistencias,
        expurgados=expurgados,
        anomalias=anomalias,
        medianas=medianas,
    ).getvalue()


def preparar_excel_download(
    base_padronizada: pd.DataFrame,
    processados: pd.DataFrame,
    inconsistencias: pd.DataFrame,
    expurgados: pd.DataFrame,
    anomalias: pd.DataFrame,
    medianas: pd.DataFrame,
) -> bytes:
    return exportar_excel(
        base_bruta=base_padronizada,
        base_validos=processados,
        inconsistencias=inconsistencias,
        expurgados=expurgados,
        anomalias=anomalias,
        medianas=medianas,
    ).getvalue()


def main() -> None:
    st.title("Luna")
    st.caption("Análise de tempos operacionais")

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

            st.caption("Arquivos configurados no Google Drive:")
            if available_files:
                st.write(available_files)
            else:
                st.warning("Nenhum arquivo foi configurado em [drive_files] nos secrets.")

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
                "Base para cálculo da janela",
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
        st.session_state["assinatura_processamento"] = assinatura_atual
        st.session_state["ultima_unidade_processada"] = unidade
        st.session_state.pop("excel_bytes", None)
        st.session_state.pop("zip_bytes", None)

    tab_base, tab_validacao, tab_processamento, tab_cliente, tab_resultados, tab_exportacao = st.tabs(
        ["Base", "Validação", "Processamento", "Painel do Cliente", "Resultados", "Exportação"]
    )

    if "assinatura_processamento" not in st.session_state:
        with tab_base:
            st.info("Selecione a unidade, ajuste os parâmetros e clique em 'Processar base'.")
        return

    if st.session_state.get("assinatura_processamento") != assinatura_atual:
        st.sidebar.warning("Há alterações de unidade/parâmetros ainda não processadas.")

    try:
        with st.spinner("Processando base..."):
            dados_processados = processar_base(
                unidade=st.session_state["assinatura_processamento"]["unidade"],
                tempo_min_expurgo=st.session_state["assinatura_processamento"]["tempo_min_expurgo"],
                tempo_max_anomalia=st.session_state["assinatura_processamento"]["tempo_max_anomalia"],
                eventos_previos=st.session_state["assinatura_processamento"]["eventos_previos"],
                minimo_apontamentos=st.session_state["assinatura_processamento"]["minimo_apontamentos"],
                tempo_padrao_poucos_apontamentos=st.session_state["assinatura_processamento"]["tempo_padrao_poucos_apontamentos"],
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

    janelas_entrega = pd.DataFrame()
    if st.session_state["assinatura_processamento"].get("exibir_janelas_entrega", False):
        janelas_entrega = calcular_janelas_entrega(
            processados=processados,
            cobertura=st.session_state["assinatura_processamento"]["cobertura_janela"] / 100,
            usar_coluna=st.session_state["assinatura_processamento"]["base_janela"],
        )

    with tab_base:
        st.subheader("Visão da base")

        col1, col2, col3 = st.columns(3)
        col1.metric("Linhas brutas", len(base_bruta))
        col2.metric("Colunas encontradas", relatorio_validacao["total_columns"])
        col3.metric("Estrutura válida", "Sim" if relatorio_validacao["is_valid"] else "Não")

        exibir_preview_df(base_padronizada, "Pré-visualização da base", limite=100, height=450)

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

            st.stop()

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

        exibir_preview_df(
            pd.DataFrame(relatorio_validacao["mapping_preview"]),
            "Mapeamento aplicado",
            limite=1000,
            height=320,
        )

        st.subheader("Schema oficial")
        st.dataframe(
            get_schema_dataframe(),
            use_container_width=True,
            height=260,
        )

        exibir_preview_df(
            get_aliases_dataframe(),
            "Aliases reconhecidos",
            limite=1000,
            height=320,
        )

    with tab_processamento:
        st.subheader("KPIs do processamento")

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.metric("Linhas válidas", kpis.get("linhas_validas", 0))
        col2.metric("Inconsistências", kpis.get("inconsistencias", 0))
        col3.metric("Expurgados", kpis.get("expurgados", 0))
        col4.metric("Anomalias", kpis.get("anomalias", 0))
        col5.metric("Clientes únicos", kpis.get("clientes_unicos", 0))
        col6.metric("Mediana global", kpis.get("mediana_global_fmt", "00:00:00"))

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
            }
        )

        if st.session_state["assinatura_processamento"].get("exibir_janelas_entrega", False):
            st.subheader("Parâmetros das janelas")
            st.write(
                {
                    "exibir_janelas_entrega": "Sim",
                    "cobertura_janela": f"{st.session_state['assinatura_processamento']['cobertura_janela']}%",
                    "base_janela": st.session_state["assinatura_processamento"]["base_janela"],
                    "qtd_janelas_geradas": len(janelas_entrega),
                }
            )

    with tab_cliente:
        st.subheader("Evolução de tempos por cliente")

        if processados is None or processados.empty:
            st.info("Não há dados processados para exibir o painel do cliente.")
        else:
            clientes_disponiveis = sorted(
                processados["Cod_Cliente"].astype(str).str.strip().dropna().unique().tolist()
            )

            if not clientes_disponiveis:
                st.info("Não há clientes disponíveis após o processamento.")
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
                        "Mostrar rótulos",
                        value=True,
                        key="mostrar_rotulos_tempo_cliente",
                    )

                dados_cliente, resumo_cliente = montar_dados_cliente(
                    processados=processados,
                    medianas=medianas,
                    cliente=cliente,
                )

                janela_cliente = pd.DataFrame()
                if not janelas_entrega.empty:
                    janela_cliente = janelas_entrega[
                        janelas_entrega["Cod_Cliente"].astype(str).str.strip() == str(cliente).strip()
                    ].copy()

                if dados_cliente.empty:
                    st.warning("Não há dados válidos para o cliente selecionado.")
                else:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Qtd. entregas", resumo_cliente["qtd_entregas"])
                    col2.metric("Média de Vol_caixas", resumo_cliente["media_vol_caixas_fmt"])
                    col3.metric("Mediana do tempo", resumo_cliente["mediana_tempo_fmt"])

                    if resumo_cliente["qtd_invalidos_convertidos"] > 0:
                        st.warning(
                            f"Foram encontrados {resumo_cliente['qtd_invalidos_convertidos']} registros de volume "
                            f"que não puderam ser convertidos corretamente e foram exibidos como 0 no gráfico."
                        )

                    st.caption(
                        "Validação de Vol_caixas — "
                        f"Total: {resumo_cliente['qtd_total_volume']} | "
                        f"Zeros reais: {resumo_cliente['qtd_zeros_reais']} | "
                        f"Nulos na origem: {resumo_cliente['qtd_nulos_origem']} | "
                        f"Inválidos convertidos: {resumo_cliente['qtd_invalidos_convertidos']}"
                    )

                    grafico = criar_grafico_cliente(
                        dados_cliente=dados_cliente,
                        resumo=resumo_cliente,
                        mostrar_rotulos_tempo=mostrar_rotulos_tempo,
                    )
                    st.plotly_chart(grafico, use_container_width=True)

                    if st.session_state["assinatura_processamento"].get("exibir_janelas_entrega", False):
                        st.subheader("Janela de entrega do cliente")

                        base_janela_ativa = st.session_state["assinatura_processamento"]["base_janela"]

                        if janela_cliente.empty:
                            st.warning(
                                "Não foi possível localizar a janela calculada para este cliente. "
                                "O gráfico abaixo será exibido somente com as aberturas/finalizações."
                            )

                            grafico_aberturas = criar_grafico_aberturas_cliente(
                                dados_cliente=dados_cliente,
                                janela_cliente=pd.DataFrame(),
                                base_janela=base_janela_ativa,
                            )
                            st.plotly_chart(grafico_aberturas, use_container_width=True)
                        else:
                            linha_janela = janela_cliente.iloc[0]

                            cj1, cj2, cj3, cj4, cj5 = st.columns(5)
                            cj1.metric("Início da janela", linha_janela["Janela_Inicio"])
                            cj2.metric("Fim da janela", linha_janela["Janela_Fim"])
                            cj3.metric("Largura", f"{int(round(float(linha_janela['Janela_Largura_Min'])))} min")
                            cj4.metric("Cobertura real", linha_janela["Cobertura_Real"])
                            cj5.metric("Cruza meia-noite", linha_janela["Cruza_MeiaNoite"])

                            st.caption(
                                f"Base utilizada: {linha_janela['Base_Janela']} | "
                                f"Período predominante: {linha_janela['Periodo_Predominante']}"
                            )

                            grafico_aberturas = criar_grafico_aberturas_cliente(
                                dados_cliente=dados_cliente,
                                janela_cliente=janela_cliente,
                                base_janela=base_janela_ativa,
                            )
                            st.plotly_chart(grafico_aberturas, use_container_width=True)

                    colunas_tabela = [
                        "DataHora_Entrega_Label",
                        "Hora_Abertura",
                        "Hora_Finalizacao",
                        "Tempo_Formatado",
                        "Vol_caixas_num",
                    ]
                    if "tour_display_id" in dados_cliente.columns:
                        colunas_tabela.append("tour_display_id")

                    exibicao_cliente = dados_cliente[colunas_tabela].copy()

                    rename_map = {
                        "DataHora_Entrega_Label": "Data da entrega",
                        "Hora_Abertura": "Abertura",
                        "Hora_Finalizacao": "Finalização",
                        "Tempo_Formatado": "Tempo gasto",
                        "Vol_caixas_num": "Vol_caixas",
                    }
                    if "tour_display_id" in exibicao_cliente.columns:
                        rename_map["tour_display_id"] = "Tour"

                    exibicao_cliente = exibicao_cliente.rename(columns=rename_map)

                    linha_resumo = {
                        "Data da entrega": "Média/Mediana",
                        "Abertura": "-",
                        "Finalização": "-",
                        "Tempo gasto": resumo_cliente["mediana_tempo_fmt"],
                        "Vol_caixas": resumo_cliente["media_vol_caixas"],
                    }
                    if "Tour" in exibicao_cliente.columns:
                        linha_resumo["Tour"] = "-"

                    exibicao_cliente = pd.concat(
                        [exibicao_cliente, pd.DataFrame([linha_resumo])],
                        ignore_index=True,
                    )

                    st.dataframe(exibicao_cliente, use_container_width=True, height=360)

    with tab_resultados:
        exibir_preview_df(medianas, "Medianas por cliente", limite=1000, height=320)

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
                "Base_Janela",
            ]
            exibir_preview_df(
                janelas_entrega[colunas_janela_exibir] if not janelas_entrega.empty else janelas_entrega,
                "Janelas de entrega",
                limite=1000,
                height=320,
            )

        exibir_preview_df(inconsistencias, "Inconsistências", limite=1000, height=240)
        exibir_preview_df(expurgados, "Expurgados", limite=1000, height=240)
        exibir_preview_df(anomalias, "Anomalias", limite=1000, height=240)

    with tab_exportacao:
        st.subheader("Exportação")
        st.caption("Download rápido: ZIP com CSVs")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Preparar ZIP rápido (CSV)", use_container_width=True):
                with st.spinner("Gerando ZIP com CSVs..."):
                    st.session_state["zip_bytes"] = preparar_zip_download(
                        base_padronizada=base_padronizada,
                        processados=processados,
                        inconsistencias=inconsistencias,
                        expurgados=expurgados,
                        anomalias=anomalias,
                        medianas=medianas,
                    )

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
                    st.session_state["excel_bytes"] = preparar_excel_download(
                        base_padronizada=base_padronizada,
                        processados=processados,
                        inconsistencias=inconsistencias,
                        expurgados=expurgados,
                        anomalias=anomalias,
                        medianas=medianas,   
                    )

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
