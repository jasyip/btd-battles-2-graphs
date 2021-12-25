from enum import Enum, auto

class DataFileType(Enum):
    EXCEL  = auto()
    CSV    = auto()
    JSON   = auto()
    PICKLE = auto()

class ConfigFileType(Enum):
    YAML   = {"yml", "yaml"}
    SHELVE = {"*"}
