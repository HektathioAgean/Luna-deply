from io import BytesIO
import zipfile

import pandas as pd


FATOR_IQR_PADRAO = 1.5

COLUNAS_MEDIANAS_CLIENTE = [
    "Cod_Cliente",
    "Qtd_Apontamentos",
    "Qtd_Base_Calculo",
    "Qtd_Base_Limpa_Boxplot",
    "Qtd_Outliers_Boxplot",
    "Mediana_Tempo_Sec",
    "Mediana_Tempo_Formatada",
    "Tempo_Ideal_Q1_Sec",
    "Tempo_Ideal_Q1_Formatado",
    "Diferenca_Mediana_Q1_Sec",
    "Diferenca_Mediana_Q1_Formatada",
    "Diferenca_Mediana_Q1_Percentual",
    "Q1_Sec",
    "Q3_Sec",
    "IQR_Sec",
    "Limite_Inferior_Boxplot_Sec",
    "Limite_Superior_Boxplot_Sec",
    "Metodo_Aplicado",
    "Metodo_Ideal_Aplicado",
]


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


def aplicar_ajuste_percentual(valor: float | int | None, ajuste_percentual: float) -> int:
    """
    Aplica ajuste percentual sobre um tempo em segundos.
    """
    if valor is None or pd.isna(valor):
        return 0

    valor_float = float(valor)
    return int(round(valor_float + (valor_float * (float(ajuste_percentual) / 100))))


def calcular_boxplot_iqr_tempos(
    serie: pd.Series,
    fator_iqr: float = FATOR_IQR_PADRAO,
) -> dict:
    """
    Calcula Q1, Q3, IQR e remove outliers pelo criterio classico de boxplot.

    Regras:
    - Q1 = percentil 25%
    - Q3 = percentil 75%
    - IQR = Q3 - Q1
    - limite inferior = Q1 - fator_iqr * IQR
    - limite superior = Q3 + fator_iqr * IQR
    - outlier = valor fora do intervalo [limite inferior, limite superior]

    Se IQR = 0, a serie original e mantida.
    """
    tempos = pd.to_numeric(serie, errors="coerce").dropna()

    if tempos.empty:
        return {
            "q1": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
            "limite_inferior": 0.0,
            "limite_superior": 0.0,
            "tempos_limpos": tempos,
            "qtd_base": 0,
            "qtd_base_limpa": 0,
            "qtd_outliers": 0,
            "metodo": "sem_tempos_validos",
        }

    q1 = float(tempos.quantile(0.25))
    q3 = float(tempos.quantile(0.75))
    iqr = float(q3 - q1)

    if iqr == 0:
        return {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "limite_inferior": float(tempos.min()),
            "limite_superior": float(tempos.max()),
            "tempos_limpos": tempos,
            "qtd_base": int(len(tempos)),
            "qtd_base_limpa": int(len(tempos)),
            "qtd_outliers": 0,
            "metodo": "q1_sem_outlier_iqr_zero",
        }

    limite_inferior = q1 - (float(fator_iqr) * iqr)
    limite_superior = q3 + (float(fator_iqr) * iqr)

    mask_valido = tempos.between(limite_inferior, limite_superior, inclusive="both")
    tempos_limpos = tempos.loc[mask_valido]
    qtd_outliers = int((~mask_valido).sum())

    if tempos_limpos.empty:
        tempos_limpos = tempos
        metodo = "q1_fallback_base_original_boxplot_vazio"
        qtd_outliers = 0
    else:
        metodo = "q1_boxplot_iqr"

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "limite_inferior": float(limite_inferior),
        "limite_superior": float(limite_superior),
        "tempos_limpos": tempos_limpos,
        "qtd_base": int(len(tempos)),
        "qtd_base_limpa": int(len(tempos_limpos)),
        "qtd_outliers": int(qtd_outliers),
        "metodo": metodo,
    }


def calcular_medianas_por_cliente(
    df: pd.DataFrame,
    eventos_previos: int,
    minimo_apontamentos: int,
    tempo_padrao_poucos_apontamentos: int,
    ajuste_percentual: float,
) -> pd.DataFrame:
    """
    Calcula os tempos de referencia por cliente.

    Saidas principais:
    - Mediana_Tempo_Sec: tempo realista, baseado na mediana dos eventos considerados.
    - Tempo_Ideal_Q1_Sec: tempo ideal, baseado no 1o quartil apos limpeza por boxplot/IQR.
    - Diferenca_Mediana_Q1_Sec: oportunidade entre tempo realista e tempo ideal.
    """
    columns = COLUNAS_MEDIANAS_CLIENTE

    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    dados = df.copy()
    dados["Cod_Cliente"] = dados["Cod_Cliente"].astype(str).str.strip()
    dados = dados[dados["Cod_Cliente"] != ""].copy()

    if dados.empty:
        return pd.DataFrame(columns=columns)

    dados["Chegou_em"] = pd.to_datetime(dados["Chegou_em"], errors="coerce")
    dados["Tempo_Sec"] = pd.to_numeric(dados["Tempo_Sec"], errors="coerce")
    dados = dados.dropna(subset=["Chegou_em", "Tempo_Sec"]).copy()

    if dados.empty:
        return pd.DataFrame(columns=columns)

    dados = dados.sort_values(by="Chegou_em", ascending=False)
    contagem = dados["Cod_Cliente"].value_counts()
    resultados = []

    for cliente, grupo in dados.groupby("Cod_Cliente", sort=True):
        qtd = int(contagem.get(cliente, 0))
        grupo = grupo.sort_values(by="Chegou_em", ascending=False).copy()

        if qtd < minimo_apontamentos:
            mediana_base = int(tempo_padrao_poucos_apontamentos)
            tempo_ideal_q1_base = int(tempo_padrao_poucos_apontamentos)
            qtd_base_calculo = int(len(grupo))
            qtd_base_limpa = int(len(grupo))
            qtd_outliers = 0
            q1 = float(tempo_padrao_poucos_apontamentos)
            q3 = float(tempo_padrao_poucos_apontamentos)
            iqr = 0.0
            limite_inferior = float(tempo_padrao_poucos_apontamentos)
            limite_superior = float(tempo_padrao_poucos_apontamentos)
            metodo = "tempo_padrao_poucos_apontamentos"
            metodo_ideal = "tempo_padrao_poucos_apontamentos"
        else:
            base = grupo.head(eventos_previos) if qtd > eventos_previos else grupo
            tempos_base = pd.to_numeric(base["Tempo_Sec"], errors="coerce").dropna()

            if tempos_base.empty:
                mediana_base = 0
                tempo_ideal_q1_base = 0
                qtd_base_calculo = 0
                qtd_base_limpa = 0
                qtd_outliers = 0
                q1 = 0.0
                q3 = 0.0
                iqr = 0.0
                limite_inferior = 0.0
                limite_superior = 0.0
                metodo = "sem_tempos_validos"
                metodo_ideal = "sem_tempos_validos"
            else:
                estat = calcular_boxplot_iqr_tempos(tempos_base)
                tempos_limpos = estat["tempos_limpos"]

                mediana_base = int(round(float(tempos_base.median())))
                tempo_ideal_q1_base = int(round(float(tempos_limpos.quantile(0.25))))

                qtd_base_calculo = int(estat["qtd_base"])
                qtd_base_limpa = int(estat["qtd_base_limpa"])
                qtd_outliers = int(estat["qtd_outliers"])
                q1 = float(estat["q1"])
                q3 = float(estat["q3"])
                iqr = float(estat["iqr"])
                limite_inferior = float(estat["limite_inferior"])
                limite_superior = float(estat["limite_superior"])

                metodo = "mediana_ultimos_n" if qtd > eventos_previos else "mediana_total"
                metodo_ideal = estat["metodo"]

        mediana_ajustada = aplicar_ajuste_percentual(mediana_base, ajuste_percentual)
        tempo_ideal_q1_ajustado = aplicar_ajuste_percentual(tempo_ideal_q1_base, ajuste_percentual)

        diferenca_sec = int(max(0, mediana_ajustada - tempo_ideal_q1_ajustado))
        diferenca_percentual = (
            round((diferenca_sec / mediana_ajustada) * 100, 2) if mediana_ajustada > 0 else 0.0
        )

        resultados.append(
            {
                "Cod_Cliente": cliente,
                "Qtd_Apontamentos": qtd,
                "Qtd_Base_Calculo": qtd_base_calculo,
                "Qtd_Base_Limpa_Boxplot": qtd_base_limpa,
                "Qtd_Outliers_Boxplot": qtd_outliers,
                "Mediana_Tempo_Sec": mediana_ajustada,
                "Mediana_Tempo_Formatada": format_seconds(mediana_ajustada),
                "Tempo_Ideal_Q1_Sec": tempo_ideal_q1_ajustado,
                "Tempo_Ideal_Q1_Formatado": format_seconds(tempo_ideal_q1_ajustado),
                "Diferenca_Mediana_Q1_Sec": diferenca_sec,
                "Diferenca_Mediana_Q1_Formatada": format_seconds(diferenca_sec),
                "Diferenca_Mediana_Q1_Percentual": diferenca_percentual,
                "Q1_Sec": round(q1, 2),
                "Q3_Sec": round(q3, 2),
                "IQR_Sec": round(iqr, 2),
                "Limite_Inferior_Boxplot_Sec": round(limite_inferior, 2),
                "Limite_Superior_Boxplot_Sec": round(limite_superior, 2),
                "Metodo_Aplicado": metodo,
                "Metodo_Ideal_Aplicado": metodo_ideal,
            }
        )

    if not resultados:
        return pd.DataFrame(columns=columns)

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
    """
    Consolida KPIs principais.
    """
    mediana_global = 0.0
    tempo_ideal_q1_global = 0.0
    gap_global = 0.0
    outliers_boxplot = 0

    if medianas is not None and not medianas.empty:
        mediana_global = float(
            pd.to_numeric(medianas.get("Mediana_Tempo_Sec"), errors="coerce").median()
        )

        if "Tempo_Ideal_Q1_Sec" in medianas.columns:
            tempo_ideal_q1_global = float(
                pd.to_numeric(medianas["Tempo_Ideal_Q1_Sec"], errors="coerce").median()
            )

        if "Diferenca_Mediana_Q1_Sec" in medianas.columns:
            gap_global = float(
                pd.to_numeric(medianas["Diferenca_Mediana_Q1_Sec"], errors="coerce").median()
            )

        if "Qtd_Outliers_Boxplot" in medianas.columns:
            outliers_boxplot = int(
                pd.to_numeric(medianas["Qtd_Outliers_Boxplot"], errors="coerce")
                .fillna(0)
                .sum()
            )

    clientes_unicos = 0
    if (
        base_validos is not None
        and not base_validos.empty
        and "Cod_Cliente" in base_validos.columns
    ):
        clientes_unicos = int(base_validos["Cod_Cliente"].astype(str).nunique())

    return {
        "linhas_brutas": int(len(base_bruta)) if base_bruta is not None else 0,
        "linhas_validas": int(len(base_validos)) if base_validos is not None else 0,
        "inconsistencias": int(len(inconsistencias)) if inconsistencias is not None else 0,
        "expurgados": int(len(expurgados)) if expurgados is not None else 0,
        "anomalias": int(len(anomalias)) if anomalias is not None else 0,
        "clientes_unicos": clientes_unicos,
        "mediana_global_seg": mediana_global,
        "mediana_global_fmt": format_seconds(mediana_global),
        "tempo_ideal_q1_global_seg": tempo_ideal_q1_global,
        "tempo_ideal_q1_global_fmt": format_seconds(tempo_ideal_q1_global),
        "gap_mediana_q1_global_seg": gap_global,
        "gap_mediana_q1_global_fmt": format_seconds(gap_global),
        "outliers_boxplot": outliers_boxplot,
    }


def exportar_excel(
    base_bruta: pd.DataFrame,
    base_validos: pd.DataFrame,
    inconsistencias: pd.DataFrame,
    expurgados: pd.DataFrame,
    anomalias: pd.DataFrame,
    medianas: pd.DataFrame,
    janelas_atendimento: pd.DataFrame | None = None,
    base_detalhada: pd.DataFrame | None = None,
) -> BytesIO:
    """
    Gera arquivo Excel em memoria.

    Abas produzidas:
    - Base Bruta
    - Base Validada
    - Base Detalhada  (processados com Mediana e Tempo Ideal Q1 por linha)
    - Inconsistencias
    - Expurgados
    - Anomalias
    - Medianas Cliente
    - Janelas Atendimento
    """
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        (base_bruta if base_bruta is not None else pd.DataFrame()).to_excel(
            writer, index=False, sheet_name="Base Bruta"
        )
        (base_validos if base_validos is not None else pd.DataFrame()).to_excel(
            writer, index=False, sheet_name="Base Validada"
        )
        (base_detalhada if base_detalhada is not None else pd.DataFrame()).to_excel(
            writer, index=False, sheet_name="Base Detalhada"
        )
        (inconsistencias if inconsistencias is not None else pd.DataFrame()).to_excel(
            writer, index=False, sheet_name="Inconsistencias"
        )
        (expurgados if expurgados is not None else pd.DataFrame()).to_excel(
            writer, index=False, sheet_name="Expurgados"
        )
        (anomalias if anomalias is not None else pd.DataFrame()).to_excel(
            writer, index=False, sheet_name="Anomalias"
        )
        (medianas if medianas is not None else pd.DataFrame()).to_excel(
            writer, index=False, sheet_name="Medianas Cliente"
        )
        (janelas_atendimento if janelas_atendimento is not None else pd.DataFrame()).to_excel(
            writer, index=False, sheet_name="Janelas Atendimento"
        )

    output.seek(0)
    return output


def _df_para_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None:
        df = pd.DataFrame()
    return df.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")


def exportar_zip_csv(
    base_bruta: pd.DataFrame,
    base_validos: pd.DataFrame,
    inconsistencias: pd.DataFrame,
    expurgados: pd.DataFrame,
    anomalias: pd.DataFrame,
    medianas: pd.DataFrame,
    janelas_atendimento: pd.DataFrame | None = None,
    base_detalhada: pd.DataFrame | None = None,
) -> BytesIO:
    """
    Gera ZIP em memoria contendo os CSVs de saida.

    Arquivos produzidos:
    - base_bruta.csv
    - base_validada.csv
    - base_detalhada.csv  (processados com Mediana e Tempo Ideal Q1 por linha)
    - inconsistencias.csv
    - expurgados.csv
    - anomalias.csv
    - medianas_cliente.csv
    - janelas_atendimento.csv
    """
    output = BytesIO()

    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("base_bruta.csv", _df_para_csv_bytes(base_bruta))
        zf.writestr("base_validada.csv", _df_para_csv_bytes(base_validos))
        zf.writestr("base_detalhada.csv", _df_para_csv_bytes(base_detalhada))
        zf.writestr("inconsistencias.csv", _df_para_csv_bytes(inconsistencias))
        zf.writestr("expurgados.csv", _df_para_csv_bytes(expurgados))
        zf.writestr("anomalias.csv", _df_para_csv_bytes(anomalias))
        zf.writestr("medianas_cliente.csv", _df_para_csv_bytes(medianas))
        zf.writestr("janelas_atendimento.csv", _df_para_csv_bytes(janelas_atendimento))

    output.seek(0)
    return output
