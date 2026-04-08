import numpy as np
import pandas as pd


def transform_base(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transforma a base bruta em:
    - dados válidos
    - inconsistências

    Regras:
    - converte colunas datetime
    - trata Cod_Cliente
    - calcula Tempo_Sec
    - cria colunas derivadas
    - classifica inconsistências
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    dados = df.copy()

    # =========================
    # Tratamento de cliente
    # =========================
    dados["Cod_Cliente"] = (
        dados["Cod_Cliente"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )

    # =========================
    # Conversão de datetime
    # =========================
    dados["Chegou_em"] = pd.to_datetime(
        dados["Chegou_em"],
        errors="coerce",
        cache=True,
    )
    dados["Finalizada_em"] = pd.to_datetime(
        dados["Finalizada_em"],
        errors="coerce",
        cache=True,
    )

    # =========================
    # Tempo em segundos
    # =========================
    dados["Tempo_Sec"] = (
        dados["Finalizada_em"] - dados["Chegou_em"]
    ).dt.total_seconds()

    # =========================
    # Colunas derivadas
    # =========================
    dados["Data_Chegada"] = dados["Chegou_em"].dt.normalize()
    dados["Hora_Chegada"] = dados["Chegou_em"].dt.strftime("%H:%M:%S")
    dados["Data_Finalizacao"] = dados["Finalizada_em"].dt.normalize()
    dados["Hora_Finalizacao"] = dados["Finalizada_em"].dt.strftime("%H:%M:%S")

    dados["Ano"] = dados["Chegou_em"].dt.year.astype("Int16")
    dados["Mes"] = dados["Chegou_em"].dt.month.astype("Int8")
    dados["Dia"] = dados["Chegou_em"].dt.day.astype("Int8")
    dados["Semana"] = dados["Chegou_em"].dt.isocalendar().week.astype("Int16")
    dados["Dia_Semana"] = dados["Chegou_em"].dt.dayofweek.astype("Int8")

    # =========================
    # Regras de inconsistência
    # =========================
    motivo_datetime_invalido = dados["Chegou_em"].isna() | dados["Finalizada_em"].isna()
    motivo_cliente_vazio = dados["Cod_Cliente"].isna()
    motivo_tempo_negativo = dados["Tempo_Sec"] < 0
    motivo_tempo_nulo = dados["Tempo_Sec"].isna()

    motivos = (
        np.where(motivo_datetime_invalido, "datetime_invalido | ", "")
        + np.where(motivo_cliente_vazio, "cliente_vazio | ", "")
        + np.where(motivo_tempo_negativo, "tempo_negativo | ", "")
        + np.where(motivo_tempo_nulo, "tempo_nulo | ", "")
    )

    dados["Motivo_Inconsistencia"] = (
        pd.Series(motivos, index=dados.index, dtype="string")
        .str.rstrip(" |")
        .fillna("")
    )

    # =========================
    # Separação final
    # =========================
    inconsistencias = dados.loc[dados["Motivo_Inconsistencia"] != ""].copy()
    dados_validos = dados.loc[dados["Motivo_Inconsistencia"] == ""].copy()

    return dados_validos, inconsistencias


def aplicar_regras_operacionais(
    df: pd.DataFrame,
    tempo_min_expurgo: int,
    tempo_max_anomalia: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Separa a base válida em:
    - processados
    - expurgados
    - anomalias
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    dados = df.copy()

    mask_expurgados = dados["Tempo_Sec"] < tempo_min_expurgo
    mask_anomalias = dados["Tempo_Sec"] > tempo_max_anomalia
    mask_remover = mask_expurgados | mask_anomalias

    expurgados = dados.loc[mask_expurgados].copy()
    anomalias = dados.loc[mask_anomalias].copy()
    processados = dados.loc[~mask_remover].copy()

    return processados, expurgados, anomalias
