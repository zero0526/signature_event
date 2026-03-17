from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Dict, Set
import os
import base64

def get_env_file() -> str:
    if "ENV_FILE" in os.environ:
        return os.environ["ENV_FILE"]
    return "dev.env"

class BaseConfig(BaseSettings):
    _NOUN_POS: Set= Field(default= {"N", "Np", "Nc", "Nu", "Ny", "Nb"})
    _VERB_POS: Set= {"V", "Vb"}
    _ADV_POS: Set= {"R"} # Phó từ (đã, đang, sẽ, cố_tình...)
    _ADJ_POS: Set= {"A"} # Tính từ (ẩu, nhanh, mạnh...)
    _PREP_POS: Set= {"E"} # Giới từ (vào, ra, khỏi, xuống...)
    _CONNECTOR_POS: Set= {"L", "E", "CH"}  
    _COORD_DEP: Set= {"coord", "conj"} 
    
    class Config:
        env_file = get_env_file()
        env_file_encoding = "utf-8"


