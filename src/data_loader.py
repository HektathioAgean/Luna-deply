from io import BytesIO

import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def normalize_unit_name(unit_name: str) -> str:
    if unit_name is None:
        raise ValueError("A unidade não foi informada.")

    unit_name = str(unit_name).strip().upper()

    if not unit_name:
        raise ValueError("A unidade está vazia.")

    return unit_name


@st.cache_resource
def get_drive_service():
    if "gcp_service_account" not in st.secrets:
        raise ValueError(
            "Secret 'gcp_service_account' não encontrado. "
            "Configure os Secrets no Streamlit Cloud ou no .streamlit/secrets.toml."
        )

    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=SCOPES,
    )

    return build("drive", "v3", credentials=creds)


def get_drive_file_id(unit_name: str) -> str:
    unit_name = normalize_unit_name(unit_name)

    if "drive_files" not in st.secrets:
        raise ValueError(
            "Secret 'drive_files' não encontrado. "
            "Adicione o bloco [drive_files] nos Secrets."
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
    if "drive_files" not in st.secrets:
        return []

    return sorted([str(key) for key in st.secrets["drive_files"].keys()])


def _get_file_metadata(file_id: str) -> dict:
    service = get_drive_service()
    metadata = service.files().get(
        fileId=file_id,
        fields="id,name,mimeType",
        supportsAllDrives=True,
    ).execute()
    return metadata


def _download_file_bytes(file_id: str) -> BytesIO:
    service = get_drive_service()

    request = service.files().get_media(
        fileId=file_id,
        supportsAllDrives=True,
    )
    buffer = BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    buffer.seek(0)
    return buffer


def _read_csv_buffer(buffer: BytesIO) -> pd.DataFrame:
    tentativas = [
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "latin1"},
        {"sep": ",", "encoding": "latin1"},
        {"sep": ";", "encoding": "cp1252"},
        {"sep": ",", "encoding": "cp1252"},
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


def _read_excel_buffer(buffer: BytesIO) -> pd.DataFrame:
    try:
        buffer.seek(0)
        return pd.read_excel(buffer)
    except Exception as exc:
        raise ValueError(f"Erro ao ler arquivo Excel: {exc}") from exc


@st.cache_data(ttl=1800, show_spinner=False)
def load_unit_file(unit_name: str) -> pd.DataFrame:
    unit_name = normalize_unit_name(unit_name)
    file_id = get_drive_file_id(unit_name)

    metadata = _get_file_metadata(file_id)
    file_name = str(metadata.get("name", "")).lower()
    mime_type = str(metadata.get("mimeType", "")).lower()

    buffer = _download_file_bytes(file_id)

    if file_name.endswith(".csv") or "csv" in mime_type:
        return _read_csv_buffer(buffer)

    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return _read_excel_buffer(buffer)

    raise ValueError(
        f"Formato não suportado para a unidade '{unit_name}'. "
        f"Arquivo encontrado: '{metadata.get('name')}' | mimeType: '{metadata.get('mimeType')}'. "
        "Use CSV, XLSX ou XLS."
    )
