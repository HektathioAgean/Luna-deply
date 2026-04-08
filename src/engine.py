from io import BytesIO
from pathlib import Path

import pandas as pd

from config import EXPORT_DIR


def format_seconds(value: float | int | None) -> str:
    """
    Converte segundos em HH:MM:SS.
    """
    if value is None or pd.isna(value):
        return "00:00:00"

    total = int(round(float(value)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_seconds_hhmm(value: float | int | None) -> str:
    """
    Converte segundos em HH:MM.
    """
    if value is None or pd.isna(value):
        return "00:00"

    total = int(round(float(value)))
    hours = total // 3600
    minutes = (total % 3600) // 60

    return f"{hours:02d}:{minutes:02d}"


def calcular_medianas_por_cliente(
    df: pd.DataFrame,
    eventos_previos: int,
    minimo_apontamentos: int,
    tempo_padrao_poucos_apontamentos: int,
    ajuste_percentual: float,
) -> pd.DataFrame:
    """
    Calcula mediana por cliente com regras operacionais.
    """
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


def montar_evolucao_cliente(
    base_validos: pd.DataFrame,
    medianas: pd.DataFrame,
    cliente: str,
) -> tuple[pd.DataFrame, float]:
    """
    Monta a evolução do tempo de um cliente específico.

    Retorna:
    - dataframe consolidado por data de chegada
    - mediana geral do cliente
    """
    columns = [
        "Data_Chegada",
        "Tempo_Sec",
        "Tempo_HHMM",
        "Tempo_HHMMSS",
        "Cod_Cliente",
        "Tipo",
    ]

    if base_validos is None or base_validos.empty:
        return pd.DataFrame(columns=columns), 0.0

    if "Cod_Cliente" not in base_validos.columns:
        return pd.DataFrame(columns=columns), 0.0

    if "Data_Chegada" not in base_validos.columns or "Tempo_Sec" not in base_validos.columns:
        return pd.DataFrame(columns=columns), 0.0

    cliente = str(cliente).strip()

    dados = base_validos.copy()
    dados["Cod_Cliente"] = dados["Cod_Cliente"].astype(str).str.strip()
    dados = dados[dados["Cod_Cliente"] == cliente].copy()

    if dados.empty:
        return pd.DataFrame(columns=columns), 0.0

    dados["Data_Chegada"] = pd.to_datetime(dados["Data_Chegada"], errors="coerce")
    dados["Tempo_Sec"] = pd.to_numeric(dados["Tempo_Sec"], errors="coerce")
    dados = dados.dropna(subset=["Data_Chegada", "Tempo_Sec"]).copy()

    if dados.empty:
        return pd.DataFrame(columns=columns), 0.0

    evolucao = (
        dados.groupby("Data_Chegada", as_index=False)["Tempo_Sec"]
        .median()
        .sort_values("Data_Chegada")
        .reset_index(drop=True)
    )

    mediana_cliente = 0.0
    if medianas is not None and not medianas.empty:
        med = medianas.copy()
        med["Cod_Cliente"] = med["Cod_Cliente"].astype(str).str.strip()
        linha = med.loc[med["Cod_Cliente"] == cliente, "Mediana_Tempo_Sec"]
        if not linha.empty:
            mediana_cliente = float(linha.iloc[0])

    if mediana_cliente == 0.0 and not evolucao.empty:
        mediana_cliente = float(evolucao["Tempo_Sec"].median())

    evolucao["Tempo_HHMM"] = evolucao["Tempo_Sec"].apply(format_seconds_hhmm)
    evolucao["Tempo_HHMMSS"] = evolucao["Tempo_Sec"].apply(format_seconds)
    evolucao["Cod_Cliente"] = cliente
    evolucao["Tipo"] = "Entrega"

    linha_mediana = pd.DataFrame(
        [
            {
                "Data_Chegada": pd.NaT,
                "Tempo_Sec": mediana_cliente,
                "Tempo_HHMM": format_seconds_hhmm(mediana_cliente),
                "Tempo_HHMMSS": format_seconds(mediana_cliente),
                "Cod_Cliente": cliente,
                "Tipo": "Mediana",
            }
        ]
    )

    evolucao_final = pd.concat([evolucao, linha_mediana], ignore_index=True)
    evolucao_final = evolucao_final[columns]

    return evolucao_final, mediana_cliente


def build_kpis(
    base_bruta: pd.DataFrame,
    base_validos: pd.DataFrame,
    inconsistencias: pd.DataFrame,
    expurgados: pd.DataFrame,
    anomalias: pd.DataFrame,
    medianas: pd.DataFrame,
) -> dict:
    """
    Consolida KPIs principais.
    """
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
    evolucao_cliente: pd.DataFrame | None = None,
) -> BytesIO:
    """
    Gera arquivo Excel em memória.
    """
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        base_bruta.to_excel(writer, index=False, sheet_name="Base Bruta")
        base_validos.to_excel(writer, index=False, sheet_name="Base Validada")
        inconsistencias.to_excel(writer, index=False, sheet_name="Inconsistencias")
        expurgados.to_excel(writer, index=False, sheet_name="Expurgados")
        anomalias.to_excel(writer, index=False, sheet_name="Anomalias")
        medianas.to_excel(writer, index=False, sheet_name="Medianas Cliente")

        if evolucao_cliente is not None and not evolucao_cliente.empty:
            export_df = evolucao_cliente.copy()
            if "Data_Chegada" in export_df.columns:
                export_df["Data_Chegada"] = export_df["Data_Chegada"].astype("string")
            export_df.to_excel(writer, index=False, sheet_name="Evolucao Cliente")

    output.seek(0)
    return output


def salvar_excel_em_disco(
    base_bruta: pd.DataFrame,
    base_validos: pd.DataFrame,
    inconsistencias: pd.DataFrame,
    expurgados: pd.DataFrame,
    anomalias: pd.DataFrame,
    medianas: pd.DataFrame,
    evolucao_cliente: pd.DataFrame | None = None,
    file_name: str = "luna_resultado.xlsx",
) -> Path:
    """
    Salva o Excel consolidado no diretório exports.
    """
    buffer = exportar_excel(
        base_bruta=base_bruta,
        base_validos=base_validos,
        inconsistencias=inconsistencias,
        expurgados=expurgados,
        anomalias=anomalias,
        medianas=medianas,
        evolucao_cliente=evolucao_cliente,
    )

    file_path = EXPORT_DIR / file_name

    with open(file_path, "wb") as f:
        f.write(buffer.getvalue())

    return file_path
