from io import BytesIO

import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def normalize_unit_name(unit_name: str) -> str:
    """
    Padroniza o nome da unidade.
    Ex.: 'mga' -> 'MGA'
    """
    if unit_name is None:
        raise ValueError("A unidade não foi informada.")

    unit_name = str(unit_name).strip().upper()

    if not unit_name:
        raise ValueError("A unidade está vazia.")

    return unit_name


@st.cache_resource
def get_drive_service():
    """
    Cria e reutiliza a conexão autenticada com o Google Drive.
    """
    if "gcp_service_account" not in st.secrets:
        raise ValueError(
            "Secret 'gcp_service_account' não encontrado. "
            "Configure o arquivo .streamlit/secrets.toml."
        )

    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=SCOPES,
    )

    return build("drive", "v3", credentials=creds)


def get_drive_file_id(unit_name: str) -> str:
    """
    Busca o file_id da unidade no bloco [drive_files] do secrets.
    """
    unit_name = normalize_unit_name(unit_name)

    if "drive_files" not in st.secrets:
        raise ValueError(
            "Secret 'drive_files' não encontrado. "
            "Adicione o bloco [drive_files] no .streamlit/secrets.toml."
        )

    drive_files = st.secrets["drive_files"]

    if unit_name not in drive_files:
        available = ", ".join(sorted(drive_files.keys())) if drive_files else "nenhum"
        raise FileNotFoundError(
            f"Arquivo da unidade '{unit_name}' não foi configurado em [drive_files]. "
            f"Unidades disponíveis: {available}"
        )

    file_id = str(drive_files[unit_name]).strip()

    if not file_id:
        raise ValueError(f"O file_id da unidade '{unit_name}' está vazio.")

    return file_id


def list_available_unit_files() -> list[str]:
    """
    Lista as unidades disponíveis com base nos secrets.
    """
    if "drive_files" not in st.secrets:
        return []

    return sorted([str(key) for key in st.secrets["drive_files"].keys()])


def _download_file_bytes(file_id: str) -> BytesIO:
    """
    Baixa o arquivo do Google Drive para memória.
    """
    service = get_drive_service()

    request = service.files().get_media(fileId=file_id)
    buffer = BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    buffer.seek(0)
    return buffer


def _read_csv_buffer(buffer: BytesIO) -> pd.DataFrame:
    """
    Lê CSV com algumas tentativas de encoding e separador.
    """
    tentativas = [
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "latin1"},
        {"sep": ",", "encoding": "latin1"},
    ]

    erros = []

    for tentativa in tentativas:
        try:
            buffer.seek(0)
            df = pd.read_csv(
                buffer,
                sep=tentativa["sep"],
                encoding=tentativa["encoding"],
                low_memory=False,
            )
            if not df.empty and len(df.columns) > 1:
                return df
        except Exception as exc:
            erros.append(
                f"sep={tentativa['sep']} | encoding={tentativa['encoding']} | erro={exc}"
            )

    raise ValueError(
        "Não foi possível ler o CSV com as combinações testadas. "
        + " | ".join(erros)
    )


@st.cache_data(ttl=1800)
def load_unit_file(unit_name: str) -> pd.DataFrame:
    """
    Carrega o CSV da unidade a partir do Google Drive.
    """
    unit_name = normalize_unit_name(unit_name)
    file_id = get_drive_file_id(unit_name)
    buffer = _download_file_bytes(file_id)
    return _read_csv_buffer(buffer)
