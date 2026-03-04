# real-time eeg streaming with brainflow - supports synthetic, openbci, muse, etc

from .boards import BoardConfig, SUPPORTED_BOARDS, get_board_info, list_boards
from .client import BrainFlowClient, StreamingError, stream_to_callback, quick_record
from .recorder import StreamRecorder, ContinuousRecorder, record_session

__all__ = [
    # Client
    "BrainFlowClient",
    "StreamingError",
    "stream_to_callback",
    "quick_record",
    # Boards
    "BoardConfig",
    "SUPPORTED_BOARDS",
    "get_board_info",
    "list_boards",
    # Recording
    "StreamRecorder",
    "ContinuousRecorder",
    "record_session",
]
