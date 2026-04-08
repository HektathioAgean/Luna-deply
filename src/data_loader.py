from pathlib import Path

import pandas as pd

from config import DATA_DIR


def normalize_unit_name(unit_name: str) -> str:
    if unit_name is None:
        raise ValueError("A unidade não foi informada.")

    unit_name = str(unit_name).strip().upper()

    if not unit_name:
        raise ValueError("A unidade está vazia.")

    return unit_name


def get_unit_file_candidates(unit_name: str) -> list[Path]:
    """
    Prioriza CSV e depois XLSX.
    """
    unit_name = normalize_unit_name(unit_name)

    return [
        DATA_DIR / f"{unit_name}_data.csv",
        DATA_DIR / f"{unit_name}_data.xlsx",
    ]


def list_available_unit_files() -> list[str]:
    """
    Lista arquivos disponíveis na pasta data nos padrões:
    - *_data.csv
    - *_data.xlsx
    """
    if not DATA_DIR.exists():
        return []

    files = []
    files.extend([file.name for file in DATA_DIR.glob("*_data.csv")])
    files.extend([file.name for file in DATA_DIR.glob("*_data.xlsx")])

    return sorted(files)


def read_csv_safely(file_path: Path) -> pd.DataFrame:
    """
    Tenta ler CSV com combinações comuns de separador e encoding.
    """
    attempts = [
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "latin1"},
        {"sep": ",", "encoding": "latin1"},
        {"sep": ";", "encoding": "cp1252"},
        {"sep": ",", "encoding": "cp1252"},
    ]

    last_error = None

    for params in attempts:
        try:
            return pd.read_csv(file_path, low_memory=False, **params)
        except Exception as exc:
            last_error = exc

    raise ValueError(
        f"Erro ao ler o CSV '{file_path.name}'. "
        f"Não foi possível interpretar o arquivo com os formatos testados. "
        f"Último erro: {last_error}"
    )


def load_unit_file(unit_name: str) -> pd.DataFrame:
    """
    Carrega o arquivo da unidade a partir da pasta data.
    Prioridade:
    1. {UNIDADE}_data.csv
    2. {UNIDADE}_data.xlsx
    """
    candidates = get_unit_file_candidates(unit_name)

    selected_file = None
    for file_path in candidates:
        if file_path.exists():
            selected_file = file_path
            break

    if selected_file is None:
        available_files = list_available_unit_files()
        available_text = ", ".join(available_files) if available_files else "nenhum arquivo encontrado"

        expected_names = ", ".join([path.name for path in candidates])

        raise FileNotFoundError(
            f"Arquivo não encontrado para a unidade '{normalize_unit_name(unit_name)}'. "
            f"Nomes esperados: {expected_names}. "
            f"Pasta pesquisada: {DATA_DIR}. "
            f"Arquivos disponíveis: {available_text}"
        )

    try:
        if selected_file.suffix.lower() == ".csv":
            return read_csv_safely(selected_file)

        if selected_file.suffix.lower() == ".xlsx":
            return pd.read_excel(selected_file)

        raise ValueError(f"Formato não suportado: {selected_file.suffix}")

    except Exception as exc:
        raise ValueError(
            f"Erro ao ler o arquivo '{selected_file.name}': {exc}"
        ) from exc