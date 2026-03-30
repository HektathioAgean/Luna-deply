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
) -> BytesIO:
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        base_bruta.to_excel(writer, index=False, sheet_name="Base Bruta")
        base_validos.to_excel(writer, index=False, sheet_name="Base Validada")
        inconsistencias.to_excel(writer, index=False, sheet_name="Inconsistencias")
        expurgados.to_excel(writer, index=False, sheet_name="Expurgados")
        anomalias.to_excel(writer, index=False, sheet_name="Anomalias")
        medianas.to_excel(writer, index=False, sheet_name="Medianas Cliente")

    output.seek(0)
    return output


def exportar_zip_csv(
    base_bruta: pd.DataFrame,
    base_validos: pd.DataFrame,
    inconsistencias: pd.DataFrame,
    expurgados: pd.DataFrame,
    anomalias: pd.DataFrame,
    medianas: pd.DataFrame,
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
    file_name: str = "luna_resultado.xlsx",
) -> Path:
    buffer = exportar_excel(
        base_bruta=base_bruta,
        base_validos=base_validos,
        inconsistencias=inconsistencias,
        expurgados=expurgados,
        anomalias=anomalias,
        medianas=medianas,
    )

    file_path = EXPORT_DIR / file_name

    with open(file_path, "wb") as f:
        f.write(buffer.getvalue())

    return file_path
