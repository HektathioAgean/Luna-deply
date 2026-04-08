from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
ASSETS_DIR = BASE_DIR / "assets"
EXPORT_DIR = BASE_DIR / "exports"
DATA_DIR = BASE_DIR / "data"

APP_NAME = "Luna"
APP_TITLE = "Luna | Análise de Tempos"
LAYOUT = "wide"

THEME = {
    "bg": "#e0e0e0",
    "panel": "#d6d6d6",
    "text": "#000000",
    "accent": "#c0a24b",
    "chart": ["#c0a24b", "#555555", "#b0b0b0", "#1a1a1a"],
}

AVAILABLE_UNITS = [
    "MGA",
    "GPV",
    "PG",
    "NP",
]


def initialize_directories() -> None:
    """
    Cria os diretórios necessários para o projeto.
    """
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


initialize_directories()
