import re

import numpy as np
import pandas as pd


PREFERENCIA_DATAS_AMBIGUAS = "AUTO"
PREFERENCIA_PADRAO_FALLBACK = "DMY"
FORMATOS_YMD = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
]
FORMATOS_MDY = [
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
    "%m-%d-%Y %H:%M:%S",
    "%m-%d-%Y %H:%M",
    "%m-%d-%Y",
]
FORMATOS_DMY = [
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%d-%m-%Y",
]
REGEX_DATA_DIA_MES_ANO = re.compile(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?\s*$")


def _serie_vazia_datetime(index: pd.Index) -> pd.Series:
    return pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")


def _normalizar_texto_datetime(serie: pd.Series) -> pd.Series:
    serie_texto = serie.astype("string").str.strip()
    return serie_texto.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
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


def inferir_preferencia_datas_ambiguas(
    serie: pd.Series,
    preferencia_padrao: str = PREFERENCIA_PADRAO_FALLBACK,
) -> str:
    """
    Infere se a série tende a estar em DMY ou MDY.

    Regra:
    - se o primeiro bloco passar de 12, só pode ser DMY
    - se o segundo bloco passar de 12, só pode ser MDY
    - se tudo continuar ambíguo, usa o fallback
    """
    if serie is None or len(serie) == 0:
        return preferencia_padrao

    serie_texto = _normalizar_texto_datetime(serie)
    amostra = serie_texto.dropna().astype(str)

    evidencias_dmy = 0
    evidencias_mdy = 0

    for valor in amostra.head(5000):
        match = REGEX_DATA_DIA_MES_ANO.match(valor)
        if not match:
            continue

        primeiro = int(match.group(1))
        segundo = int(match.group(2))

        if primeiro > 12 and segundo <= 12:
            evidencias_dmy += 1
        elif segundo > 12 and primeiro <= 12:
            evidencias_mdy += 1

    if evidencias_dmy > evidencias_mdy:
        return "DMY"
    if evidencias_mdy > evidencias_dmy:
        return "MDY"
    return preferencia_padrao



def parse_datetime_flexivel(
    serie: pd.Series,
    preferencia_datas_ambiguas: str = PREFERENCIA_DATAS_AMBIGUAS,
) -> pd.Series:
    """
    Converte datas de forma determinística para evitar troca entre mês e dia.

    Estratégia:
    - preserva datetimes já reconhecidos
    - prioriza ISO/YMD
    - infere automaticamente DMY/MDY quando a série contém datas ambíguas
    - usa DMY como fallback final, aderente ao padrão operacional exibido no Luna
    """
    if serie is None:
        return pd.Series(dtype="datetime64[ns]")

    if pd.api.types.is_datetime64_any_dtype(serie):
        return pd.to_datetime(serie, errors="coerce")

    origem = _normalizar_texto_datetime(serie)
    resultado = _serie_vazia_datetime(serie.index)

    # 1) formatos não ambíguos / ISO
    resultado = _aplicar_formatos(resultado, origem, FORMATOS_YMD)

    ordem_recebida = str(preferencia_datas_ambiguas or "AUTO").strip().upper()
    if ordem_recebida == "AUTO":
        ordem = inferir_preferencia_datas_ambiguas(origem, preferencia_padrao=PREFERENCIA_PADRAO_FALLBACK)
    else:
        ordem = ordem_recebida

    formatos_preferidos = FORMATOS_DMY if ordem == "DMY" else FORMATOS_MDY
    formatos_alternativos = FORMATOS_MDY if ordem == "DMY" else FORMATOS_DMY

    # 2) formatos ambíguos com preferência inferida
    resultado = _aplicar_formatos(resultado, origem, formatos_preferidos)
    resultado = _aplicar_formatos(resultado, origem, formatos_alternativos)

    # 3) fallback final para formatos fora do padrão
    mask_pendente = resultado.isna() & origem.notna()
    if mask_pendente.any():
        fallback = pd.to_datetime(
            origem.loc[mask_pendente],
            errors="coerce",
            dayfirst=(ordem == "DMY"),
        )
        resultado.loc[mask_pendente] = fallback

    return resultado



def transform_base(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transforma a base bruta em:
    - dados válidos
    - inconsistências

    Regras:
    - converte colunas datetime com parse determinístico
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
    dados["Chegou_em"] = parse_datetime_flexivel(
        dados["Chegou_em"],
        preferencia_datas_ambiguas=PREFERENCIA_DATAS_AMBIGUAS,
    )
    dados["Finalizada_em"] = parse_datetime_flexivel(
        dados["Finalizada_em"],
        preferencia_datas_ambiguas=PREFERENCIA_DATAS_AMBIGUAS,
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
