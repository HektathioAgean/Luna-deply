from io import BytesIO
from pathlib import Path
import zipfile

import pandas as pd

from config import EXPORT_DIR


def format_seconds(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "00:00:00"

    total = int(round(float(value)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_minutes(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "00:00"

    total_minutes = int(round(float(value))) % (24 * 60)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


def _serie_datetime_para_minutos(serie: pd.Series) -> pd.Series:
    serie_dt = pd.to_datetime(serie, errors="coerce")
    return (
        (serie_dt.dt.hour.fillna(0) * 60)
        + serie_dt.dt.minute.fillna(0)
        + (serie_dt.dt.second.fillna(0) / 60.0)
    )


def _normalizar_janela_circular(valores: pd.Series) -> tuple[pd.Series, bool]:
    serie = pd.to_numeric(valores, errors="coerce").dropna().astype(float)

    if serie.empty:
        return serie, False

    span_direto = float(serie.max() - serie.min())
    cruza_meia_noite = span_direto > 720

    if cruza_meia_noite:
        serie = serie.where(serie >= 720, serie + 1440)

    return serie, cruza_meia_noite


def _rotulo_periodo_por_minutos(minutos: float) -> str:
    minuto = float(minutos) % (24 * 60)

    if 8 * 60 <= minuto <= 18 * 60:
        return "Comercial"
    if 6 * 60 <= minuto < 12 * 60:
        return "Diurno"
    if 12 * 60 <= minuto < 18 * 60:
        return "Vespertino"
    return "Noturno"


def calcular_medianas_por_cliente(
    df: pd.DataFrame,
    eventos_previos: int,
    minimo_apontamentos: int,
    tempo_padrao_poucos_apontamentos: int,
    ajuste_percentual: float,
) -> pd.DataFrame:
    columns = [
        "Cod_Cliente",
        "Qtd_Apontamentos",
        "Mediana_Tempo_Sec",
        "Mediana_Tempo_Formatada",
        "Metodo_Aplicado",
    ]

    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    dados = df.copy()
    dados = dados.sort_values(by="Chegou_em", ascending=False)

    contagem = dados["Cod_Cliente"].value_counts()
    resultados = []

    for cliente, grupo in dados.groupby("Cod_Cliente", sort=True):
        qtd = int(contagem.get(cliente, 0))
        grupo = grupo.sort_values(by="Chegou_em", ascending=False).copy()

        if qtd < minimo_apontamentos:
            mediana = int(tempo_padrao_poucos_apontamentos)
            metodo = "tempo_padrao_poucos_apontamentos"
        else:
            base = grupo.head(eventos_previos) if qtd > eventos_previos else grupo
            mediana = int(base["Tempo_Sec"].median()) if not base.empty else 0
            metodo = "mediana_ultimos_n" if qtd > eventos_previos else "mediana_total"

        mediana_ajustada = int(mediana + (mediana * (ajuste_percentual / 100)))

        resultados.append(
            {
                "Cod_Cliente": cliente,
                "Qtd_Apontamentos": qtd,
                "Mediana_Tempo_Sec": mediana_ajustada,
                "Mediana_Tempo_Formatada": format_seconds(mediana_ajustada),
                "Metodo_Aplicado": metodo,
            }
        )

    return (
        pd.DataFrame(resultados, columns=columns)
        .sort_values(by=["Mediana_Tempo_Sec", "Cod_Cliente"], ascending=[True, True])
        .reset_index(drop=True)
    )


def calcular_janelas_entrega(
    df: pd.DataFrame,
    cobertura_janela: float = 0.80,
    minimo_apontamentos: int = 4,
) -> pd.DataFrame:
    columns = [
        "Cod_Cliente",
        "N_Entregas",
        "Cobertura_Alvo",
        "Percentil_Inicio",
        "Percentil_Fim",
        "Ji_Minutos",
        "Jf_Minutos",
        "Ji",
        "Jf",
        "Amplitude_Horas",
        "Cruza_MeiaNoite",
        "Periodo",
        "Metodo_Janela",
    ]

    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    cobertura = float(cobertura_janela)
    cobertura = min(max(cobertura, 0.50), 0.99)

    percentil_inicio = (1 - cobertura) / 2
    percentil_fim = 1 - percentil_inicio

    dados = df.copy()
    dados = dados.dropna(subset=["Cod_Cliente", "Chegou_em", "Finalizada_em"])

    if dados.empty:
        return pd.DataFrame(columns=columns)

    dados["Minutos_Chegada"] = _serie_datetime_para_minutos(dados["Chegou_em"])
    dados["Minutos_Finalizacao"] = _serie_datetime_para_minutos(dados["Finalizada_em"])

    resultados = []

    for cliente, grupo in dados.groupby("Cod_Cliente", sort=True):
        qtd = int(len(grupo))
        chegada_norm, cruza_inicio = _normalizar_janela_circular(grupo["Minutos_Chegada"])
        fim_norm, cruza_fim = _normalizar_janela_circular(grupo["Minutos_Finalizacao"])

        if chegada_norm.empty or fim_norm.empty:
            continue

        ji_norm = float(chegada_norm.quantile(percentil_inicio))
        jf_norm = float(fim_norm.quantile(percentil_fim))

        ji_min = ji_norm % 1440
        jf_min = jf_norm % 1440

        amplitude = jf_norm - ji_norm
        if amplitude < 0:
            amplitude += 1440

        centro = (ji_norm + (amplitude / 2)) % 1440
        cruza = bool(cruza_inicio or cruza_fim or (ji_min > jf_min))
        metodo = "percentis_total" if qtd >= minimo_apontamentos else "percentis_base_reduzida"

        resultados.append(
            {
                "Cod_Cliente": cliente,
                "N_Entregas": qtd,
                "Cobertura_Alvo": f"{int(round(cobertura * 100))}%",
                "Percentil_Inicio": f"P{int(round(percentil_inicio * 100))}",
                "Percentil_Fim": f"P{int(round(percentil_fim * 100))}",
                "Ji_Minutos": round(ji_min, 2),
                "Jf_Minutos": round(jf_min, 2),
                "Ji": format_minutes(ji_min),
                "Jf": format_minutes(jf_min),
                "Amplitude_Horas": round(amplitude / 60.0, 2),
                "Cruza_MeiaNoite": "Sim" if cruza else "Não",
                "Periodo": _rotulo_periodo_por_minutos(centro),
                "Metodo_Janela": metodo,
            }
        )

    if not resultados:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(resultados, columns=columns)
        .sort_values(by=["Ji_Minutos", "Cod_Cliente"], ascending=[True, True])
        .reset_index(drop=True)
    )


def build_kpis(
    base_bruta: pd.DataFrame,
    base_validos: pd.DataFrame,
    inconsistencias: pd.DataFrame,
    expurgados: pd.DataFrame,
    anomalias: pd.DataFrame,
    medianas: pd.DataFrame,
) -> dict:
    mediana_global = 0.0
    if medianas is not None and not medianas.empty:
        mediana_global = float(medianas["Mediana_Tempo_Sec"].median())

    return {
        "linhas_brutas": int(len(base_bruta)) if base_bruta is not None else 0,
        "linhas_validas": int(len(base_validos)) if base_validos is not None else 0,
        "inconsistencias": int(len(inconsistencias)) if inconsistencias is not None else 0,
        "expurgados": int(len(expurgados)) if expurgados is not None else 0,
        "anomalias": int(len(anomalias)) if anomalias is not None else 0,
        "clientes_unicos": int(base_validos["Cod_Cliente"].nunique()) if base_validos is not None and not base_validos.empty else 0,
        "mediana_global_seg": mediana_global,
        "mediana_global_fmt": format_seconds(mediana_global),
    }


def exportar_excel(
    base_bruta: pd.DataFrame,
    base_validos: pd.DataFrame,
    inconsistencias: pd.DataFrame,
    expurgados: pd.DataFrame,
    anomalias: pd.DataFrame,
    medianas: pd.DataFrame,
    janelas: pd.DataFrame | None = None,
) -> BytesIO:
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        base_bruta.to_excel(writer, index=False, sheet_name="Base Bruta")
        base_validos.to_excel(writer, index=False, sheet_name="Base Validada")
        inconsistencias.to_excel(writer, index=False, sheet_name="Inconsistencias")
        expurgados.to_excel(writer, index=False, sheet_name="Expurgados")
        anomalias.to_excel(writer, index=False, sheet_name="Anomalias")
        medianas.to_excel(writer, index=False, sheet_name="Medianas Cliente")
        if janelas is not None:
            janelas.to_excel(writer, index=False, sheet_name="Janelas Entrega")

    output.seek(0)
    return output


def exportar_zip_csv(
    base_bruta: pd.DataFrame,
    base_validos: pd.DataFrame,
    inconsistencias: pd.DataFrame,
    expurgados: pd.DataFrame,
    anomalias: pd.DataFrame,
    medianas: pd.DataFrame,
    janelas: pd.DataFrame | None = None,
) -> BytesIO:
    output = BytesIO()

    arquivos = {
        "base_bruta.csv": base_bruta,
        "base_validada.csv": base_validos,
        "inconsistencias.csv": inconsistencias,
        "expurgados.csv": expurgados,
        "anomalias.csv": anomalias,
        "medianas_cliente.csv": medianas,
    }

    if janelas is not None:
        arquivos["janelas_entrega.csv"] = janelas

    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for nome_arquivo, df in arquivos.items():
            csv_bytes = df.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
            zf.writestr(nome_arquivo, csv_bytes)

    output.seek(0)
    return output


def salvar_excel_em_disco(
    base_bruta: pd.DataFrame,
    base_validos: pd.DataFrame,
    inconsistencias: pd.DataFrame,
    expurgados: pd.DataFrame,
    anomalias: pd.DataFrame,
    medianas: pd.DataFrame,
    janelas: pd.DataFrame | None = None,
    file_name: str = "luna_resultado.xlsx",
) -> Path:
    buffer = exportar_excel(
        base_bruta=base_bruta,
        base_validos=base_validos,
        inconsistencias=inconsistencias,
        expurgados=expurgados,
        anomalias=anomalias,
        medianas=medianas,
        janelas=janelas,
    )

    file_path = EXPORT_DIR / file_name

    with open(file_path, "wb") as f:
        f.write(buffer.getvalue())

    return file_path
