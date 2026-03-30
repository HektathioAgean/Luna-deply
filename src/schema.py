import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd


def normalize_text(value: Any) -> str:
    """
    Normaliza texto para comparação.
    """
    text = str(value or "").strip().lower()
    text = text.replace("\ufeff", "")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def slugify_column(value: Any) -> str:
    """
    Cria chave simplificada para matching de colunas.
    """
    text = normalize_text(value)
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9_ ]+", "", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


OFFICIAL_SCHEMA = {
    "Cod_Cliente": {
        "required": True,
        "dtype": "string",
        "category": "identificacao",
        "description": "Código do cliente",
    },
    "Chegou_em": {
        "required": True,
        "dtype": "datetime",
        "category": "tempo",
        "description": "Data e hora de chegada",
    },
    "Finalizada_em": {
        "required": True,
        "dtype": "datetime",
        "category": "tempo",
        "description": "Data e hora de finalização",
    },
    "tour_display_id": {
        "required": False,
        "dtype": "string",
        "category": "rota",
        "description": "Identificador do tour",
    },
    "within_radius": {
        "required": False,
        "dtype": "boolean",
        "category": "qualidade",
        "description": "Indicador dentro do raio",
    },
    "Latitude": {
        "required": False,
        "dtype": "float",
        "category": "geo",
        "description": "Latitude",
    },
    "Longitude": {
        "required": False,
        "dtype": "float",
        "category": "geo",
        "description": "Longitude",
    },
    "Motorista": {
        "required": False,
        "dtype": "string",
        "category": "operacao",
        "description": "Motorista",
    },
    "Mapa": {
        "required": False,
        "dtype": "string",
        "category": "operacao",
        "description": "Mapa/romaneio",
    },
    "Veiculo": {
        "required": False,
        "dtype": "string",
        "category": "operacao",
        "description": "Veículo",
    },
    "Transportadora": {
        "required": False,
        "dtype": "string",
        "category": "operacao",
        "description": "Transportadora",
    },
}

COLUMN_ALIASES = {
    # cliente
    "cod_cliente": "Cod_Cliente",
    "codigo_cliente": "Cod_Cliente",
    "codigo do cliente": "Cod_Cliente",
    "código_cliente": "Cod_Cliente",
    "código do cliente": "Cod_Cliente",
    "cod cliente": "Cod_Cliente",
    "codigo cliente": "Cod_Cliente",
    "código cliente": "Cod_Cliente",
    "cliente": "Cod_Cliente",
    "customer_id": "Cod_Cliente",
    "customer code": "Cod_Cliente",

    # chegada
    "chegou_em": "Chegou_em",
    "chegou em": "Chegou_em",
    "data_chegada": "Chegou_em",
    "data chegada": "Chegou_em",
    "data_hora_chegada": "Chegou_em",
    "data hora chegada": "Chegou_em",
    "arrived_at": "Chegou_em",
    "arrived": "Chegou_em",
    "arrived_date": "Chegou_em",

    # finalização
    "finalizada_em": "Finalizada_em",
    "finalizada em": "Finalizada_em",
    "data_finalizacao": "Finalizada_em",
    "data finalizacao": "Finalizada_em",
    "data_finalização": "Finalizada_em",
    "data finalização": "Finalizada_em",
    "data_hora_finalizacao": "Finalizada_em",
    "data hora finalizacao": "Finalizada_em",
    "data_hora_finalização": "Finalizada_em",
    "data hora finalização": "Finalizada_em",
    "finished_at": "Finalizada_em",
    "finished": "Finalizada_em",
    "finished_date": "Finalizada_em",

    # rota
    "tour_display_id": "tour_display_id",
    "tour id": "tour_display_id",
    "tour": "tour_display_id",
    "id_tour": "tour_display_id",
    "id do tour": "tour_display_id",
    "id mapa": "tour_display_id",

    # qualidade
    "within_radius": "within_radius",
    "within radius": "within_radius",
    "dentro_do_raio": "within_radius",
    "dentro do raio": "within_radius",

    # geo
    "latitude": "Latitude",
    "lat": "Latitude",
    "longitude": "Longitude",
    "long": "Longitude",
    "lng": "Longitude",
    "lon": "Longitude",

    # operação
    "motorista": "Motorista",
    "driver": "Motorista",
    "mapa": "Mapa",
    "romaneio": "Mapa",
    "veiculo": "Veiculo",
    "veículo": "Veiculo",
    "vehicle": "Veiculo",
    "transportadora": "Transportadora",
    "carrier": "Transportadora",
}

REQUIRED_COLUMNS = [
    col for col, meta in OFFICIAL_SCHEMA.items() if meta["required"]
]

OPTIONAL_COLUMNS = [
    col for col, meta in OFFICIAL_SCHEMA.items() if not meta["required"]
]


@dataclass
class ColumnMatch:
    original_name: str
    normalized_name: str
    standardized_name: str
    recognized: bool
    required: bool
    category: str
    dtype: str
    description: str


@dataclass
class SchemaReport:
    is_valid: bool
    total_columns: int
    total_required: int
    total_optional: int
    required_found: list[str]
    required_missing: list[str]
    optional_found: list[str]
    unknown_columns: list[str]
    duplicate_standardized_columns: list[str]
    mapping_preview: list[dict]


def get_standard_name(column_name: Any) -> str:
    normalized = normalize_text(column_name)
    slug = slugify_column(column_name)

    if normalized in COLUMN_ALIASES:
        return COLUMN_ALIASES[normalized]

    if slug in COLUMN_ALIASES:
        return COLUMN_ALIASES[slug]

    return str(column_name).strip()


def build_column_match(column_name: Any) -> ColumnMatch:
    original_name = str(column_name)
    normalized_name = normalize_text(original_name)
    standardized_name = get_standard_name(original_name)

    recognized = standardized_name in OFFICIAL_SCHEMA
    meta = OFFICIAL_SCHEMA.get(
        standardized_name,
        {
            "required": False,
            "dtype": "unknown",
            "category": "extra",
            "description": "Coluna não reconhecida pelo schema oficial",
        },
    )

    return ColumnMatch(
        original_name=original_name,
        normalized_name=normalized_name,
        standardized_name=standardized_name,
        recognized=recognized,
        required=bool(meta["required"]),
        category=str(meta["category"]),
        dtype=str(meta["dtype"]),
        description=str(meta["description"]),
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza nomes das colunas, preservando colisões com sufixo __dupX.
    """
    if df is None:
        return pd.DataFrame()

    if df.empty:
        return df.copy()

    rename_map: dict[str, str] = {}
    used_names: dict[str, int] = {}

    for col in df.columns:
        std_name = get_standard_name(col)

        if std_name in used_names:
            used_names[std_name] += 1
            final_name = f"{std_name}__dup{used_names[std_name]}"
        else:
            used_names[std_name] = 0
            final_name = std_name

        rename_map[col] = final_name

    return df.rename(columns=rename_map).copy()


def analyze_schema(df: pd.DataFrame) -> SchemaReport:
    if df is None:
        raise ValueError("O DataFrame não pode ser None.")

    matches = [build_column_match(col) for col in df.columns]

    standardized_names = [m.standardized_name for m in matches]
    base_names = [name.split("__dup")[0] for name in standardized_names]

    required_found = [col for col in REQUIRED_COLUMNS if col in base_names]
    required_missing = [col for col in REQUIRED_COLUMNS if col not in base_names]
    optional_found = [col for col in OPTIONAL_COLUMNS if col in base_names]

    unknown_columns = [m.original_name for m in matches if not m.recognized]

    duplicate_standardized_columns = sorted({
        name.split("__dup")[0]
        for name in standardized_names
        if "__dup" in name
    })

    return SchemaReport(
        is_valid=len(required_missing) == 0,
        total_columns=len(df.columns),
        total_required=len(REQUIRED_COLUMNS),
        total_optional=len(OPTIONAL_COLUMNS),
        required_found=required_found,
        required_missing=required_missing,
        optional_found=optional_found,
        unknown_columns=unknown_columns,
        duplicate_standardized_columns=duplicate_standardized_columns,
        mapping_preview=[asdict(m) for m in matches],
    )


def schema_report_to_dict(report: SchemaReport) -> dict:
    return asdict(report)


def suggest_missing_columns(report: SchemaReport) -> list[str]:
    suggestions = []

    for col in report.required_missing:
        if col == "Cod_Cliente":
            suggestions.append("Adicionar a coluna Cod_Cliente com o identificador do cliente.")
        elif col == "Chegou_em":
            suggestions.append("Adicionar a coluna Chegou_em com data e hora de chegada.")
        elif col == "Finalizada_em":
            suggestions.append("Adicionar a coluna Finalizada_em com data e hora de finalização.")
        else:
            suggestions.append(f"Adicionar a coluna obrigatória: {col}")

    return suggestions


def get_schema_dataframe() -> pd.DataFrame:
    rows = []
    for col, meta in OFFICIAL_SCHEMA.items():
        rows.append(
            {
                "Coluna": col,
                "Obrigatoria": meta["required"],
                "Tipo": meta["dtype"],
                "Categoria": meta["category"],
                "Descricao": meta["description"],
            }
        )
    return pd.DataFrame(rows)


def get_aliases_dataframe() -> pd.DataFrame:
    rows = []
    for alias, target in COLUMN_ALIASES.items():
        rows.append(
            {
                "Alias": alias,
                "Coluna_Padrao": target,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["Coluna_Padrao", "Alias"])
        .reset_index(drop=True)
    )


def check_minimum_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    report = analyze_schema(df)
    return report.is_valid, report.required_missing