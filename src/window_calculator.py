"""
Luna — window_calculator.py
Calculo de janelas operacionais de entrega por cliente.

Regras implementadas (memoria de calculo operacional):
  1. Pegar horarios historicos do cliente (Chegou_em / Finalizada_em)
  2. Converter para minutos do dia (08:30 -> 510, 23:50 -> 1430, 00:10 -> 10)
  3. Ordenar para entender concentracao real
  4. Calcular faixa de cobertura alvo (~80%, P10-P90 como referencia)
  5. Tratar cruzamento de meia-noite com leitura circular, nao linear
  6. Escolher a MENOR faixa valida que cobre o percentual desejado
  7. Converter de volta para HH:MM
  8. Marcar atributos: Janela_Inicio, Janela_Fim, Cruza_MeiaNoite, % cobertura

Integracao:
  - Recebe DataFrame processado pelo data_transformer (Chegou_em como datetime)
  - Usa Hora_Chegada se disponivel, senao extrai de Chegou_em
  - Retorna DataFrame de janelas por cliente
"""

import pandas as pd
import numpy as np
from typing import Optional

# -- Constantes ----------------------------------------------------------------

MINUTOS_DIA = 1440

PERIODOS_DIA = [
    ("Diurno",     7, 11),
    ("Vespertino", 12, 16),
    ("Noturno",    17, 23),
]

DEFAULT_COBERTURA_ALVO = 0.80
DEFAULT_MIN_ENTREGAS = 3


# -- Conversoes ----------------------------------------------------------------

def _ts_para_minutos(ts: pd.Timestamp) -> float:
    """Converte componente horario de timestamp em minutos desde meia-noite."""
    return ts.hour * 60 + ts.minute + ts.second / 60


def _minutos_para_hhmm(minutos: float) -> str:
    """Converte minutos desde meia-noite para string HH:MM."""
    m = int(round(minutos)) % MINUTOS_DIA
    return f"{m // 60:02d}:{m % 60:02d}"


def _classificar_periodo(hora: int) -> str:
    """Retorna o periodo do dia para uma hora (0-23)."""
    for nome, h_ini, h_fim in PERIODOS_DIA:
        if h_ini <= hora <= h_fim:
            return nome
    return "Noturno"


def _classificar_comercial(hora: int) -> bool:
    """Retorna True se a hora esta dentro do horario comercial (07:00 a 17:59)."""
    return 7 <= hora <= 17


# -- Nucleo: menor janela circular ---------------------------------------------

def _menor_janela_circular(
    minutos_ordenados: np.ndarray,
    cobertura_alvo: float,
) -> dict:
    """
    Encontra a menor faixa circular que cobre >= cobertura_alvo das entregas.

    Logica:
      - Os horarios sao tratados como pontos num relogio de 1440 minutos.
      - Para cada possivel ponto de inicio (= cada horario registrado),
        avanca circularmente ate cobrir o numero minimo de pontos.
      - Mede a amplitude (distancia circular) dessa janela.
      - Retorna a janela de menor amplitude entre todas as candidatas.

    Parametros
    ----------
    minutos_ordenados : np.ndarray
        Array de minutos (0-1439), ja ordenado.
    cobertura_alvo : float
        Fracao de entregas que a janela deve cobrir (ex: 0.80).

    Retorna
    -------
    dict com: inicio, fim, amplitude, cobertura, cruza_meianoite
    """
    n = len(minutos_ordenados)
    k = max(1, int(np.ceil(n * cobertura_alvo)))

    if k >= n:
        inicio = minutos_ordenados[0]
        fim = minutos_ordenados[-1]
        amp_linear = fim - inicio
        amp_circular = (MINUTOS_DIA - fim) + inicio
        if amp_circular < amp_linear:
            return {
                "inicio": fim,
                "fim": inicio,
                "amplitude": amp_circular,
                "cobertura": 1.0,
                "cruza_meianoite": True,
            }
        return {
            "inicio": inicio,
            "fim": fim,
            "amplitude": amp_linear,
            "cobertura": 1.0,
            "cruza_meianoite": False,
        }

    melhor = None

    for i in range(n):
        j = (i + k - 1) % n

        p_inicio = minutos_ordenados[i]
        p_fim = minutos_ordenados[j]

        if j >= i:
            amplitude = p_fim - p_inicio
            cruza = False
        else:
            amplitude = (MINUTOS_DIA - p_inicio) + p_fim
            cruza = True

        if melhor is None or amplitude < melhor["amplitude"]:
            melhor = {
                "inicio": p_inicio,
                "fim": p_fim,
                "amplitude": amplitude,
                "cobertura": round(k / n, 4),
                "cruza_meianoite": cruza,
            }

    return melhor


# -- Horario de pico -----------------------------------------------------------

def _calcular_pico(minutos: np.ndarray, faixa_min: int = 30) -> float:
    """
    Calcula horario de pico como centro da faixa de 30min mais frequente.
    """
    faixas = (minutos // faixa_min).astype(int)
    moda = pd.Series(faixas).mode()
    if len(moda) > 0:
        return float(moda.iloc[0] * faixa_min + faixa_min / 2)
    return float(np.median(minutos))


# -- Funcao principal ----------------------------------------------------------

def calcular_janelas(
    df: pd.DataFrame,
    col_cliente: str = "Cod_Cliente",
    col_chegada: str = "Chegou_em",
    col_hora_chegada: Optional[str] = "Hora_Chegada",
    cobertura_alvo: float = DEFAULT_COBERTURA_ALVO,
    min_entregas: int = DEFAULT_MIN_ENTREGAS,
) -> pd.DataFrame:
    """
    Calcula a janela operacional de entrega para cada cliente.

    Usa logica circular: testa todas as janelas possiveis que cobrem
    >= cobertura_alvo das entregas e escolhe a de menor amplitude.

    Parametros
    ----------
    df : DataFrame
        Base processada pelo data_transformer.
    col_cliente : str
        Coluna de codigo do cliente.
    col_chegada : str
        Coluna de datetime de chegada.
    col_hora_chegada : str, optional
        Coluna de hora ja extraida pelo data_transformer (Hora_Chegada).
        Se presente, usa direto. Senao, extrai de col_chegada.
    cobertura_alvo : float
        Fracao de cobertura desejada (padrao 0.80 = 80%).
    min_entregas : int
        Minimo de entregas para calcular janela.

    Retorna
    -------
    DataFrame com colunas:
        Cod_Cliente, Qtd_Entregas, Janela_Inicio, Janela_Fim,
        Amplitude_Min, Horario_Pico, Periodo_Pico, Comercial,
        Cobertura_Efetiva, Cruza_MeiaNoite,
        Inicio_Min, Fim_Min  (float, para graficos)
    """
    if col_cliente not in df.columns:
        raise ValueError(f"Coluna '{col_cliente}' nao encontrada no DataFrame.")
    if col_chegada not in df.columns:
        raise ValueError(f"Coluna '{col_chegada}' nao encontrada no DataFrame.")

    df_work = df[[col_cliente]].copy()

    if col_hora_chegada and col_hora_chegada in df.columns:
        hc = df[col_hora_chegada]
        try:
            df_work["_minutos"] = hc.apply(
                lambda x: x.hour * 60 + x.minute + x.second / 60
                if hasattr(x, "hour") else np.nan
            )
        except Exception:
            df_work["_minutos"] = np.nan
    else:
        df_work["_minutos"] = np.nan

    mask_vazio = df_work["_minutos"].isna()
    if mask_vazio.any():
        chegadas = pd.to_datetime(df.loc[mask_vazio.index, col_chegada], errors="coerce")
        df_work.loc[mask_vazio, "_minutos"] = chegadas.apply(
            lambda x: _ts_para_minutos(x) if pd.notna(x) else np.nan
        )

    df_work = df_work.dropna(subset=["_minutos"])

    resultados = []

    for cliente, grupo in df_work.groupby(col_cliente):
        n = len(grupo)
        minutos = np.sort(grupo["_minutos"].values)

        if n < min_entregas:
            resultados.append(_registro_insuficiente(cliente, n))
            continue

        janela = _menor_janela_circular(minutos, cobertura_alvo)

        pico = _calcular_pico(minutos)
        hora_pico = int(pico // 60) % 24

        resultados.append({
            "Cod_Cliente": cliente,
            "Qtd_Entregas": n,
            "Janela_Inicio": _minutos_para_hhmm(janela["inicio"]),
            "Janela_Fim": _minutos_para_hhmm(janela["fim"]),
            "Amplitude_Min": int(round(janela["amplitude"])),
            "Horario_Pico": _minutos_para_hhmm(pico),
            "Periodo_Pico": _classificar_periodo(hora_pico),
            "Comercial": _classificar_comercial(hora_pico),
            "Cobertura_Efetiva": round(janela["cobertura"] * 100, 1),
            "Cruza_MeiaNoite": janela["cruza_meianoite"],
            "Inicio_Min": round(janela["inicio"], 1),
            "Fim_Min": round(janela["fim"], 1),
        })

    df_janelas = pd.DataFrame(resultados)
    df_janelas = df_janelas.sort_values(
        "Inicio_Min", na_position="last"
    ).reset_index(drop=True)

    return df_janelas


def _registro_insuficiente(cliente, n: int) -> dict:
    """Registro para cliente com entregas insuficientes."""
    return {
        "Cod_Cliente": cliente,
        "Qtd_Entregas": n,
        "Janela_Inicio": "—",
        "Janela_Fim": "—",
        "Amplitude_Min": None,
        "Horario_Pico": "—",
        "Periodo_Pico": "—",
        "Comercial": None,
        "Cobertura_Efetiva": None,
        "Cruza_MeiaNoite": False,
        "Inicio_Min": None,
        "Fim_Min": None,
    }


# -- Metricas agregadas --------------------------------------------------------

def resumo_janelas(df_janelas: pd.DataFrame) -> dict:
    """KPIs agregados das janelas calculadas."""
    validos = df_janelas.dropna(subset=["Amplitude_Min"])

    if validos.empty:
        return {
            "total_clientes": len(df_janelas),
            "clientes_com_janela": 0,
            "clientes_sem_janela": len(df_janelas),
            "amplitude_media_min": 0,
            "amplitude_mediana_min": 0,
            "cobertura_media": 0,
            "periodo_mais_comum": "—",
            "clientes_comercial": 0,
            "clientes_meianoite": 0,
            "menor_janela_min": 0,
            "maior_janela_min": 0,
        }

    periodos = validos["Periodo_Pico"].value_counts()

    return {
        "total_clientes": len(df_janelas),
        "clientes_com_janela": len(validos),
        "clientes_sem_janela": len(df_janelas) - len(validos),
        "amplitude_media_min": int(round(validos["Amplitude_Min"].mean())),
        "amplitude_mediana_min": int(round(validos["Amplitude_Min"].median())),
        "cobertura_media": round(validos["Cobertura_Efetiva"].mean(), 1),
        "periodo_mais_comum": periodos.index[0] if len(periodos) > 0 else "—",
        "clientes_comercial": int(validos["Comercial"].sum()),
        "clientes_meianoite": int(validos["Cruza_MeiaNoite"].sum()),
        "menor_janela_min": int(validos["Amplitude_Min"].min()),
        "maior_janela_min": int(validos["Amplitude_Min"].max()),
    }


# -- Dados para grafico --------------------------------------------------------

def preparar_dados_grafico(df_janelas: pd.DataFrame, top_n: int = 40) -> list[dict]:
    """Prepara lista de dicts para renderizacao no grafico de janelas."""
    validos = df_janelas.dropna(subset=["Inicio_Min", "Fim_Min"]).head(top_n)

    dados = []
    for _, row in validos.iterrows():
        dados.append({
            "cliente": str(row["Cod_Cliente"]),
            "inicio": row["Inicio_Min"],
            "fim": row["Fim_Min"],
            "amplitude": row["Amplitude_Min"],
            "cobertura": row["Cobertura_Efetiva"],
            "periodo": row["Periodo_Pico"],
            "comercial": row["Comercial"],
            "pico_str": row["Horario_Pico"],
            "janela_str": f'{row["Janela_Inicio"]} - {row["Janela_Fim"]}',
            "cruza_meianoite": row["Cruza_MeiaNoite"],
        })

    return dados
