# board configs for brainflow

from dataclasses import dataclass, field
from typing import Optional

try:
    from brainflow.board_shim import BoardIds
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    # Create a mock BoardIds for when brainflow isn't installed
    class BoardIds:
        SYNTHETIC_BOARD = -1
        CYTON_BOARD = 0
        GANGLION_BOARD = 1
        CYTON_DAISY_BOARD = 2
        MUSE_2_BOARD = 22
        MUSE_S_BOARD = 21
        MUSE_2016_BOARD = 41


# Supported boards with friendly names and descriptions
SUPPORTED_BOARDS = {
    "synthetic": {
        "board_id": BoardIds.SYNTHETIC_BOARD,
        "name": "Synthetic Board",
        "description": "Simulated EEG data for testing (no hardware required)",
        "channels": 16,
        "sfreq": 250,
        "requires_serial": False,
        "requires_mac": False,
    },
    "cyton": {
        "board_id": BoardIds.CYTON_BOARD,
        "name": "OpenBCI Cyton",
        "description": "8-channel biosensing board",
        "channels": 8,
        "sfreq": 250,
        "requires_serial": True,
        "requires_mac": False,
    },
    "cyton_daisy": {
        "board_id": BoardIds.CYTON_DAISY_BOARD,
        "name": "OpenBCI Cyton + Daisy",
        "description": "16-channel biosensing board",
        "channels": 16,
        "sfreq": 125,
        "requires_serial": True,
        "requires_mac": False,
    },
    "ganglion": {
        "board_id": BoardIds.GANGLION_BOARD,
        "name": "OpenBCI Ganglion",
        "description": "4-channel biosensing board",
        "channels": 4,
        "sfreq": 200,
        "requires_serial": False,
        "requires_mac": True,
    },
    "muse_2": {
        "board_id": BoardIds.MUSE_2_BOARD,
        "name": "Muse 2",
        "description": "4-channel consumer EEG headband",
        "channels": 4,
        "sfreq": 256,
        "requires_serial": False,
        "requires_mac": True,
    },
    "muse_s": {
        "board_id": BoardIds.MUSE_S_BOARD,
        "name": "Muse S",
        "description": "4-channel sleep/meditation headband",
        "channels": 4,
        "sfreq": 256,
        "requires_serial": False,
        "requires_mac": True,
    },
}


@dataclass
class BoardConfig:
    # config for a brainflow board - holds connection params

    board_id: int
    serial_port: Optional[str] = None
    mac_address: Optional[str] = None
    ip_address: Optional[str] = None
    ip_port: int = 0
    timeout: int = 15
    streamer_params: str = ""
    board_name: str = ""

    def __post_init__(self):
        # sets board name from SUPPORTED_BOARDS if not given
        if not self.board_name:
            for name, info in SUPPORTED_BOARDS.items():
                if info["board_id"] == self.board_id:
                    self.board_name = info["name"]
                    break
            else:
                self.board_name = f"Board {self.board_id}"

    @classmethod
    def synthetic(cls) -> "BoardConfig":
        # synthetic board - no hardware needed
        return cls(
            board_id=BoardIds.SYNTHETIC_BOARD,
            board_name="Synthetic Board",
        )

    @classmethod
    def cyton(cls, serial_port: str) -> "BoardConfig":
        # openbci cyton config
        return cls(
            board_id=BoardIds.CYTON_BOARD,
            serial_port=serial_port,
            board_name="OpenBCI Cyton",
        )

    @classmethod
    def cyton_daisy(cls, serial_port: str) -> "BoardConfig":
        # cyton + daisy config
        return cls(
            board_id=BoardIds.CYTON_DAISY_BOARD,
            serial_port=serial_port,
            board_name="OpenBCI Cyton + Daisy",
        )

    @classmethod
    def ganglion(cls, mac_address: Optional[str] = None, serial_port: Optional[str] = None) -> "BoardConfig":
        # ganglion config
        return cls(
            board_id=BoardIds.GANGLION_BOARD,
            mac_address=mac_address,
            serial_port=serial_port,
            board_name="OpenBCI Ganglion",
        )

    @classmethod
    def muse_2(cls, mac_address: Optional[str] = None) -> "BoardConfig":
        # muse 2 config
        return cls(
            board_id=BoardIds.MUSE_2_BOARD,
            mac_address=mac_address or "",
            board_name="Muse 2",
        )

    @classmethod
    def muse_s(cls, mac_address: Optional[str] = None) -> "BoardConfig":
        # muse s config
        return cls(
            board_id=BoardIds.MUSE_S_BOARD,
            mac_address=mac_address or "",
            board_name="Muse S",
        )

    @classmethod
    def from_name(cls, name: str, **kwargs) -> "BoardConfig":
        # create config from board name string
        name = name.lower().replace("-", "_").replace(" ", "_")

        if name not in SUPPORTED_BOARDS:
            valid_names = ", ".join(SUPPORTED_BOARDS.keys())
            raise ValueError(f"Unknown board: {name}. Valid options: {valid_names}")

        board_info = SUPPORTED_BOARDS[name]

        # Validate required parameters
        if board_info["requires_serial"] and "serial_port" not in kwargs:
            raise ValueError(f"{board_info['name']} requires serial_port parameter")

        return cls(
            board_id=board_info["board_id"],
            board_name=board_info["name"],
            **kwargs,
        )


def get_board_info(board_name: str) -> dict:
    # get info dict for a board
    name = board_name.lower().replace("-", "_").replace(" ", "_")

    if name not in SUPPORTED_BOARDS:
        valid_names = ", ".join(SUPPORTED_BOARDS.keys())
        raise ValueError(f"Unknown board: {name}. Valid options: {valid_names}")

    return SUPPORTED_BOARDS[name].copy()


def list_boards() -> list[dict]:
    # lists all supported boards
    return [
        {"key": key, **info}
        for key, info in SUPPORTED_BOARDS.items()
    ]
