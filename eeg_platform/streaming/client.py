# brainflow client for streaming eeg data

import time
from typing import Optional, Callable
import numpy as np

from .boards import BoardConfig, SUPPORTED_BOARDS

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels
    from brainflow.data_filter import DataFilter
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False


class StreamingError(Exception):
    # streaming went wrong
    pass


class BrainFlowClient:
    # high-level client for brainflow streaming

    def __init__(self, config: BoardConfig, log_level: str = "warning"):
        # init the client with board config
        if not BRAINFLOW_AVAILABLE:
            raise StreamingError(
                "BrainFlow is not installed. Install it with: pip install brainflow"
            )

        self.config = config
        self._board: Optional[BoardShim] = None
        self._is_streaming = False

        # Set log level
        log_levels = {
            "trace": LogLevels.LEVEL_TRACE,
            "debug": LogLevels.LEVEL_DEBUG,
            "info": LogLevels.LEVEL_INFO,
            "warning": LogLevels.LEVEL_WARN,
            "error": LogLevels.LEVEL_ERROR,
            "off": LogLevels.LEVEL_OFF,
        }
        BoardShim.set_log_level(log_levels.get(log_level.lower(), LogLevels.LEVEL_WARN))

        # Initialize board parameters
        self._params = self._create_params()

    def _create_params(self) -> "BrainFlowInputParams":
        # makes brainflow params from config
        params = BrainFlowInputParams()

        if self.config.serial_port:
            params.serial_port = self.config.serial_port
        if self.config.mac_address:
            params.mac_address = self.config.mac_address
        if self.config.ip_address:
            params.ip_address = self.config.ip_address
        if self.config.ip_port:
            params.ip_port = self.config.ip_port
        if self.config.timeout:
            params.timeout = self.config.timeout

        return params

    @property
    def board_id(self) -> int:
        return self.config.board_id

    @property
    def sfreq(self) -> float:
        return float(BoardShim.get_sampling_rate(self.board_id))

    @property
    def eeg_channels(self) -> list[int]:
        return BoardShim.get_eeg_channels(self.board_id)

    @property
    def n_channels(self) -> int:
        return len(self.eeg_channels)

    @property
    def channel_names(self) -> list[str]:
        # gets channel names, falls back to generic
        try:
            names = BoardShim.get_eeg_names(self.board_id)
            if names:
                return names.split(",")
        except Exception:
            pass

        # Generate generic names
        return [f"EEG{i+1}" for i in range(self.n_channels)]

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    def prepare(self) -> None:
        # connects to the device
        if self._board is not None:
            return  # Already prepared

        try:
            self._board = BoardShim(self.board_id, self._params)
            self._board.prepare_session()
        except Exception as e:
            self._board = None
            raise StreamingError(f"Failed to prepare session: {e}") from e

    def start(self, buffer_size: int = 450000) -> None:
        # starts streaming data
        if self._is_streaming:
            return  # Already streaming

        if self._board is None:
            self.prepare()

        try:
            self._board.start_stream(buffer_size, self.config.streamer_params)
            self._is_streaming = True
        except Exception as e:
            raise StreamingError(f"Failed to start stream: {e}") from e

    def stop(self) -> None:
        # stops streaming
        if not self._is_streaming:
            return

        try:
            self._board.stop_stream()
        except Exception:
            pass  # Ignore errors during stop
        finally:
            self._is_streaming = False

    def release(self) -> None:
        # releases board session - call when done
        self.stop()

        if self._board is not None:
            try:
                self._board.release_session()
            except Exception:
                pass  # Ignore errors during release
            finally:
                self._board = None

    def get_board_data(self, num_samples: Optional[int] = None) -> np.ndarray:
        # gets raw data from buffer
        if not self._is_streaming:
            raise StreamingError("Not currently streaming. Call start() first.")

        try:
            if num_samples is not None:
                return self._board.get_board_data(num_samples)
            else:
                return self._board.get_board_data()
        except Exception as e:
            raise StreamingError(f"Failed to get data: {e}") from e

    def get_current_board_data(self, num_samples: int) -> np.ndarray:
        # peeks at recent data without removing from buffer
        if not self._is_streaming:
            raise StreamingError("Not currently streaming. Call start() first.")

        try:
            return self._board.get_current_board_data(num_samples)
        except Exception as e:
            raise StreamingError(f"Failed to get current data: {e}") from e

    def get_data(self, num_samples: Optional[int] = None) -> dict:
        # gets eeg data in platform-compatible dict format
        raw_data = self.get_board_data(num_samples)

        # Extract EEG channels
        eeg_channels = self.eeg_channels
        eeg_data = raw_data[eeg_channels, :]

        n_samples = eeg_data.shape[1]
        sfreq = self.sfreq

        # Create time array
        time_arr = np.arange(n_samples, dtype=np.float64) / sfreq

        return {
            "data": eeg_data,
            "ch_names": self.channel_names,
            "sfreq": sfreq,
            "time": time_arr,
            "meta": {
                "source": "brainflow",
                "board_name": self.config.board_name,
                "board_id": self.board_id,
                "n_channels": len(eeg_channels),
                "n_samples": n_samples,
            },
        }

    def get_data_count(self) -> int:
        # how many samples in buffer
        if not self._is_streaming:
            return 0

        try:
            return self._board.get_board_data_count()
        except Exception:
            return 0

    def insert_marker(self, value: float) -> None:
        # inserts a marker into the stream
        if not self._is_streaming:
            raise StreamingError("Not currently streaming. Call start() first.")

        try:
            self._board.insert_marker(value)
        except Exception as e:
            raise StreamingError(f"Failed to insert marker: {e}") from e

    def __enter__(self) -> "BrainFlowClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def stream_to_callback(
    config: BoardConfig,
    callback: Callable[[dict], None],
    duration_sec: Optional[float] = None,
    chunk_sec: float = 0.1,
) -> None:
    # streams data to a callback - good for real-time processing
    client = BrainFlowClient(config)

    try:
        client.start()
        sfreq = client.sfreq
        chunk_samples = int(chunk_sec * sfreq)

        start_time = time.time()
        while True:
            # Check duration limit
            if duration_sec is not None:
                elapsed = time.time() - start_time
                if elapsed >= duration_sec:
                    break

            # Wait for enough samples
            while client.get_data_count() < chunk_samples:
                time.sleep(0.01)

            # Get and process data
            data = client.get_data(chunk_samples)
            callback(data)

    finally:
        client.release()


def quick_record(
    config: BoardConfig,
    duration_sec: float,
) -> dict:
    # quick way to record for a fixed duration
    with BrainFlowClient(config) as client:
        time.sleep(duration_sec)
        return client.get_data()
