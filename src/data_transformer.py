import numpy as np
import pandas as pd

from config import DATE_INPUT_ORDER


FORMATOS_ISO = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
]

FORMATOS_DMY = [
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%d-%m-%Y",
]

FORMATOS_MDY = [
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
    "%m-%d-%Y %H:%M:%S",
    "%m-%d-%Y %H:%M",
    "%m-%d-%Y",
]


def _serie_vazia_datetime(index: pd.Index) -> pd.Series:
    return pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")


def _normalizar_texto_datetime(serie: pd.Series) -> pd.Series:
    serie_texto = serie.astype("string").str.strip()
    return serie_texto.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "none": pd.NA,
            "None": pd.NA,
            "NaT": pd.NA,
            "<NA>": pd.NA,
        }
    )


def _aplicar_formatos(
    destino: pd.Series,
    origem: pd.Series,
    formatos: list[str],
) -> pd.Series:
    resultado = destino.copy()

    for formato in formatos:
        mask_pendente = resultado.isna() & origem.notna()
        if not mask_pendente.any():
            break

        parsed = pd.to_datetime(
            origem.loc[mask_pendente],
            format=formato,
            errors="coerce",
        )
        resultado.loc[mask_pendente] = parsed

    return resultado


def parse_datetime_configurada(serie: pd.Series) -> pd.Series:
    """
    Faz o parse de datas usando a configuração central do projeto.

    Regras:
    - prioriza formatos ISO/YMD
    - depois usa SOMENTE a ordem configurada em DATE_INPUT_ORDER
    - fallback final respeita dayfirst da configuração
    """
    if serie is None:
        return pd.Series(dtype="datetime64[ns]")

    if pd.api.types.is_datetime64_any_dtype(serie):
        return pd.to_datetime(serie, errors="coerce")

    origem = _normalizar_texto_datetime(serie)
    resultado = _serie_vazia_datetime(serie.index)

    # 1) ISO / YMD primeiro
    resultado = _aplicar_formatos(resultado, origem, FORMATOS_ISO)

    # 2) Formato configurado
    ordem = str(DATE_INPUT_ORDER).strip().upper()
    if ordem == "MDY":
        formatos_principais = FORMATOS_MDY
        dayfirst = False
    else:
        formatos_principais = FORMATOS_DMY
        dayfirst = True

    resultado = _aplicar_formatos(resultado, origem, formatos_principais)

    # 3) Fallback final mantendo a mesma lógica
    mask_pendente = resultado.isna() & origem.notna()
    if mask_pendente.any():
        fallback = pd.to_datetime(
            origem.loc[mask_pendente],
            errors="coerce",
            dayfirst=dayfirst,
        )
        resultado.loc[mask_pendente] = fallback

    return resultado


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
    dados["Chegou_em"] = parse_datetime_configurada(dados["Chegou_em"])
    dados["Finalizada_em"] = parse_datetime_configurada(dados["Finalizada_em"])

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
