import re

import numpy as np
import pandas as pd


FORMATOS_DATA_HORA_BR = [
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
]

FORMATOS_DATA_HORA_US = [
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
]

FORMATOS_DATA_HORA_ISO = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
]


def _detectar_prioridade_data(serie: pd.Series) -> str:
    """
    Detecta se datas com barra estão mais próximas do padrão BR ou US.

    Regra:
    - Se o primeiro bloco possuir valor > 12 em alguma linha, assume BR (dd/mm).
    - Se o segundo bloco possuir valor > 12 em alguma linha, assume US (mm/dd).
    - Se não houver evidência clara, usa US como padrão operacional.
      Isso evita casos como 03/10/2026 virar 03 de outubro quando a base vem em mm/dd/yyyy.
    """
    amostra = serie.dropna().astype(str).str.strip().head(5000)

    primeiro_maior_12 = 0
    segundo_maior_12 = 0

    padrao = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{2,4})")

    for valor in amostra:
        match = padrao.match(valor)
        if not match:
            continue

        primeiro = int(match.group(1))
        segundo = int(match.group(2))

        if primeiro > 12:
            primeiro_maior_12 += 1
        if segundo > 12:
            segundo_maior_12 += 1

    if primeiro_maior_12 > segundo_maior_12:
        return "BR"

    if segundo_maior_12 > primeiro_maior_12:
        return "US"

    return "US"


def converter_datetime_operacional(serie: pd.Series) -> pd.Series:
    """
    Converte datas sem depender de inferência automática do pandas/dateutil.

    O objetivo é reduzir conversões ambíguas e eliminar o warning:
    "Could not infer format... falling back to dateutil".

    Prioridade:
    1. Formatos ISO.
    2. Formato BR ou US conforme detecção da coluna.
    3. Formato oposto como fallback.
    """
    if serie is None:
        return pd.Series(dtype="datetime64[ns]")

    entrada = serie.copy()

    if pd.api.types.is_datetime64_any_dtype(entrada):
        return pd.to_datetime(entrada, errors="coerce")

    texto = (
        entrada.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )

    resultado = pd.Series(pd.NaT, index=texto.index, dtype="datetime64[ns]")

    def aplicar_formatos(formatos: list[str]) -> None:
        nonlocal resultado
        faltantes = resultado.isna() & texto.notna()

        for formato in formatos:
            if not faltantes.any():
                break

            convertido = pd.to_datetime(
                texto.loc[faltantes],
                format=formato,
                errors="coerce",
                cache=True,
            )

            mask_ok = convertido.notna()
            if mask_ok.any():
                idx_ok = convertido.index[mask_ok]
                resultado.loc[idx_ok] = convertido.loc[idx_ok]

            faltantes = resultado.isna() & texto.notna()

    aplicar_formatos(FORMATOS_DATA_HORA_ISO)

    prioridade = _detectar_prioridade_data(texto)
    if prioridade == "BR":
        aplicar_formatos(FORMATOS_DATA_HORA_BR)
        aplicar_formatos(FORMATOS_DATA_HORA_US)
    else:
        aplicar_formatos(FORMATOS_DATA_HORA_US)
        aplicar_formatos(FORMATOS_DATA_HORA_BR)

    return resultado


def transform_base(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transforma a base bruta em:
    - dados válidos
    - inconsistências

    Regras:
    - converte colunas datetime com formato controlado
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
    # Tratamento de identificadores opcionais
    # =========================
    for coluna in ["tour_display_id", "Tour", "Mapa"]:
        if coluna in dados.columns:
            dados[coluna] = (
                dados[coluna]
                .astype("string")
                .str.strip()
                .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
            )

    # =========================
    # Conversão de datetime
    # =========================
    dados["Chegou_em"] = converter_datetime_operacional(dados["Chegou_em"])
    dados["Finalizada_em"] = converter_datetime_operacional(dados["Finalizada_em"])

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

    dados["Tempo_Sec"] = pd.to_numeric(dados["Tempo_Sec"], errors="coerce")

    mask_expurgados = dados["Tempo_Sec"] < tempo_min_expurgo
    mask_anomalias = dados["Tempo_Sec"] > tempo_max_anomalia
    mask_remover = mask_expurgados | mask_anomalias

    expurgados = dados.loc[mask_expurgados].copy()
    anomalias = dados.loc[mask_anomalias].copy()
    processados = dados.loc[~mask_remover].copy()

    return processados, expurgados, anomalias
