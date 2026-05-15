import math
import re

import pandas as pd


MINUTOS_DIA = 1440

DIAS_SEMANA_MAP = {
    0: "SEG",
    1: "TER",
    2: "QUA",
    3: "QUI",
    4: "SEX",
    5: "SAB",
    6: "DOM",
}

OPCOES_DIA_SEMANA = ["TODOS", "SEG", "TER", "QUA", "QUI", "SEX", "SAB"]


# -- Formatacao ----------------------------------------------------------------

def formatar_numero(value: float | int | None, casas: int = 2) -> str:
    if value is None or pd.isna(value):
        return "0"
    return f"{float(value):,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def formatar_minutos_hhmm(valor: float | int | None) -> str:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return ""
    total_min = int(round(float(valor))) % MINUTOS_DIA
    horas = total_min // 60
    minutos = total_min % 60
    return f"{horas:02d}:{minutos:02d}"


# -- Tempo e periodo -----------------------------------------------------------

def minutos_desde_meia_noite(serie: pd.Series) -> pd.Series:
    """Converte serie de datetime para minutos desde meia-noite."""
    serie_dt = pd.to_datetime(serie, errors="coerce")
    return (serie_dt.dt.hour * 60) + serie_dt.dt.minute + (serie_dt.dt.second / 60.0)


def classificar_periodo(hora_media: float | int | None) -> str:
    """
    Classifica o periodo do dia a partir de minutos desde meia-noite.

    Regras:
      07:00 a 11:59 -> Diurno
      12:00 a 16:59 -> Vespertino
      17:00 a 06:59 -> Noturno
    """
    if hora_media is None or (isinstance(hora_media, float) and math.isnan(hora_media)):
        return ""
    hora = (float(hora_media) % MINUTOS_DIA) / 60
    if 7 <= hora < 12:
        return "Diurno"
    if 12 <= hora < 17:
        return "Vespertino"
    return "Noturno"


def classificar_comercial(hora_media: float | int | None) -> str:
    """
    Indica se o horario esta dentro do horario comercial.

    Regra:
      07:00 a 17:59 -> Sim
      Demais horarios -> Nao
    """
    if hora_media is None or (isinstance(hora_media, float) and math.isnan(hora_media)):
        return ""
    hora = (float(hora_media) % MINUTOS_DIA) / 60
    return "Sim" if 7 <= hora < 18 else "Nao"


# -- Janela circular -----------------------------------------------------------

def calcular_janela_circular_minima(
    minutos: list[float],
    cobertura: float = 0.80,
) -> dict:
    """
    Encontra a menor faixa circular que cobre >= cobertura das entregas.
    """
    if not minutos:
        return {
            "janela_inicio_min": None,
            "janela_fim_min": None,
            "largura_min": None,
            "cobertura_real": 0.0,
            "cruza_meia_noite": False,
        }

    minutos_limpos = sorted([float(x) % MINUTOS_DIA for x in minutos if not pd.isna(x)])
    n = len(minutos_limpos)

    if n == 0:
        return {
            "janela_inicio_min": None,
            "janela_fim_min": None,
            "largura_min": None,
            "cobertura_real": 0.0,
            "cruza_meia_noite": False,
        }

    if n == 1:
        unico = minutos_limpos[0]
        return {
            "janela_inicio_min": unico,
            "janela_fim_min": unico,
            "largura_min": 0.0,
            "cobertura_real": 1.0,
            "cruza_meia_noite": False,
        }

    qtd_cobertura = max(1, int(math.ceil(n * cobertura)))
    minutos_ext = minutos_limpos + [m + MINUTOS_DIA for m in minutos_limpos]

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
            melhor_inicio = inicio % MINUTOS_DIA
            melhor_fim = fim % MINUTOS_DIA

    cruza = bool(
        melhor_inicio is not None
        and melhor_fim is not None
        and melhor_fim < melhor_inicio
    )

    return {
        "janela_inicio_min": melhor_inicio,
        "janela_fim_min": melhor_fim,
        "largura_min": menor_largura,
        "cobertura_real": qtd_cobertura / n if n > 0 else 0.0,
        "cruza_meia_noite": cruza,
    }


# -- Volume de caixas ----------------------------------------------------------

def normalizar_numero_texto(value) -> float | None:
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    texto = str(value).strip()

    if texto == "" or texto.lower() in {"nan", "none", "<na>"}:
        return None

    texto = texto.replace("\xa0", "").replace(" ", "")
    texto = re.sub(r"[^0-9,.\-]", "", texto)

    if texto in {"", "-", ".", ","}:
        return None

    if "," in texto and "." in texto:
        if texto.rfind(",") > texto.rfind("."):
            texto = texto.replace(".", "").replace(",", ".")
        else:
            texto = texto.replace(",", "")
    elif "," in texto:
        if texto.count(",") > 1:
            ultima = texto.rfind(",")
            texto = texto[:ultima].replace(",", "") + "." + texto[ultima + 1:]
        else:
            texto = texto.replace(",", ".")
    elif "." in texto:
        if texto.count(".") > 1:
            ultima = texto.rfind(".")
            texto = texto[:ultima].replace(".", "") + "." + texto[ultima + 1:]

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

    return serie_numerica.fillna(0.0), resumo_validacao


# -- Streamlit -----------------------------------------------------------------

def preparar_dataframe_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas textuais/mistas para string para evitar erros de
    serializacao Arrow no Streamlit. Preserva colunas numericas.
    """
    if df is None:
        return pd.DataFrame()

    if isinstance(df, dict):
        df = pd.DataFrame(df)

    exibicao = df.copy()

    for coluna in exibicao.columns:
        serie = exibicao[coluna]
        if (
            pd.api.types.is_object_dtype(serie)
            or pd.api.types.is_string_dtype(serie)
            or isinstance(serie.dtype, pd.CategoricalDtype)
        ):
            exibicao[coluna] = (
                serie.astype("string")
                .replace({"nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
                .fillna("")
            )

    return exibicao
