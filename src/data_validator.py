import pandas as pd

from schema import (
    analyze_schema,
    schema_report_to_dict,
    standardize_columns,
)


def apply_schema_standardization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica a padronizacao de nomes de colunas.
    """
    return standardize_columns(df)


def build_validation_report(df: pd.DataFrame) -> dict:
    """
    Gera relatorio de validacao da estrutura.
    """
    report = analyze_schema(df)
    return schema_report_to_dict(report)
