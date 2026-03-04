# records eeg streams to files

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd

from .boards import BoardConfig
from .client import BrainFlowClient, StreamingError


class StreamRecorder:
    # records eeg streams to csv files

    def __init__(
        self,
        config: BoardConfig,
        output_dir: str | Path = "recordings",
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._client: Optional[BrainFlowClient] = None
        self._is_recording = False
        self._recorded_data: list[np.ndarray] = []
        self._start_time: Optional[float] = None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def _generate_filename(self, prefix: str = "recording") -> str:
        # makes timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.csv"

    def record(
        self,
        duration_sec: float,
        filename: Optional[str] = None,
        chunk_sec: float = 0.5,
        on_chunk: Optional[Callable[[dict, float], None]] = None,
        on_complete: Optional[Callable[[Path], None]] = None,
    ) -> Path:
        # records for the given duration, saves to csv
        if filename is None:
            filename = self._generate_filename()

        output_path = self.output_dir / filename

        self._client = BrainFlowClient(self.config)
        self._recorded_data = []
        self._is_recording = True
        self._start_time = time.time()

        try:
            self._client.start()
            sfreq = self._client.sfreq
            ch_names = self._client.channel_names
            chunk_samples = int(chunk_sec * sfreq)

            while True:
                elapsed = time.time() - self._start_time
                if elapsed >= duration_sec:
                    break

                # Wait for chunk
                while self._client.get_data_count() < chunk_samples:
                    time.sleep(0.01)
                    # Check if duration exceeded during wait
                    if time.time() - self._start_time >= duration_sec:
                        break

                # Get available data
                if self._client.get_data_count() > 0:
                    data = self._client.get_data()
                    self._recorded_data.append(data["data"])

                    if on_chunk:
                        on_chunk(data, elapsed)

            # Get any remaining data
            remaining = self._client.get_data_count()
            if remaining > 0:
                data = self._client.get_data()
                self._recorded_data.append(data["data"])

        finally:
            self._is_recording = False
            if self._client:
                self._client.release()
                self._client = None

        # Combine and save data
        if self._recorded_data:
            combined = np.hstack(self._recorded_data)
            self._save_csv(combined, ch_names, sfreq, output_path)

            # Save metadata
            self._save_metadata(output_path, sfreq, duration_sec)

            if on_complete:
                on_complete(output_path)

            return output_path
        else:
            raise StreamingError("No data recorded")

    def _save_csv(
        self,
        data: np.ndarray,
        ch_names: list[str],
        sfreq: float,
        path: Path,
    ) -> None:
        # saves data to csv
        # Create time column
        n_samples = data.shape[1]
        time_col = np.arange(n_samples) / sfreq

        # Build DataFrame
        df_dict = {"time": time_col}
        for i, ch_name in enumerate(ch_names):
            df_dict[ch_name] = data[i, :]

        df = pd.DataFrame(df_dict)
        df.to_csv(path, index=False)

    def _save_metadata(
        self,
        csv_path: Path,
        sfreq: float,
        duration_sec: float,
    ) -> None:
        # saves metadata json alongside csv
        meta_path = csv_path.with_suffix(".json")
        metadata = {
            "recording_file": csv_path.name,
            "board_name": self.config.board_name,
            "board_id": self.config.board_id,
            "sfreq": sfreq,
            "duration_sec": duration_sec,
            "recorded_at": datetime.now().isoformat(),
            "source": "brainflow",
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)


class ContinuousRecorder:
    # continuous recorder with manual start/stop

    def __init__(
        self,
        config: BoardConfig,
        output_dir: str | Path = "recordings",
        chunk_sec: float = 0.5,
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_sec = chunk_sec

        self._client: Optional[BrainFlowClient] = None
        self._is_recording = False
        self._recorded_data: list[np.ndarray] = []
        self._start_time: Optional[float] = None
        self._ch_names: list[str] = []
        self._sfreq: float = 0.0

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def elapsed_sec(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def start(self, filename: Optional[str] = None) -> None:
        # starts recording
        if self._is_recording:
            raise StreamingError("Already recording")

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.csv"

        self._output_path = self.output_dir / filename
        self._client = BrainFlowClient(self.config)
        self._recorded_data = []

        try:
            self._client.start()
            self._ch_names = self._client.channel_names
            self._sfreq = self._client.sfreq
            self._is_recording = True
            self._start_time = time.time()
        except Exception as e:
            if self._client:
                self._client.release()
                self._client = None
            raise StreamingError(f"Failed to start recording: {e}") from e

    def collect(self) -> int:
        # grabs data from buffer - call periodically to avoid overflow
        if not self._is_recording:
            raise StreamingError("Not recording")

        count = self._client.get_data_count()
        if count > 0:
            data = self._client.get_data()
            self._recorded_data.append(data["data"])
            return data["data"].shape[1]
        return 0

    def stop(self) -> Path:
        # stops recording and saves the data
        if not self._is_recording:
            raise StreamingError("Not recording")

        # Collect any remaining data
        self.collect()

        duration = self.elapsed_sec
        self._is_recording = False

        try:
            if self._client:
                self._client.release()
                self._client = None

            if not self._recorded_data:
                raise StreamingError("No data recorded")

            # Combine and save
            combined = np.hstack(self._recorded_data)
            self._save_data(combined, duration)

            return self._output_path

        finally:
            self._recorded_data = []
            self._start_time = None

    def _save_data(self, data: np.ndarray, duration: float) -> None:
        # saves the recorded data
        # Save CSV
        n_samples = data.shape[1]
        time_col = np.arange(n_samples) / self._sfreq

        df_dict = {"time": time_col}
        for i, ch_name in enumerate(self._ch_names):
            df_dict[ch_name] = data[i, :]

        df = pd.DataFrame(df_dict)
        df.to_csv(self._output_path, index=False)

        # Save metadata
        meta_path = self._output_path.with_suffix(".json")
        metadata = {
            "recording_file": self._output_path.name,
            "board_name": self.config.board_name,
            "board_id": self.config.board_id,
            "sfreq": self._sfreq,
            "n_channels": len(self._ch_names),
            "n_samples": n_samples,
            "duration_sec": duration,
            "recorded_at": datetime.now().isoformat(),
            "source": "brainflow",
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def __enter__(self) -> "ContinuousRecorder":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._is_recording:
            try:
                self.stop()
            except StreamingError:
                pass


def record_session(
    board_name: str = "synthetic",
    duration_sec: float = 10.0,
    output_dir: str | Path = "recordings",
    serial_port: Optional[str] = None,
    mac_address: Optional[str] = None,
    verbose: bool = True,
) -> Path:
    # convenience func to record a session
    # Build config
    kwargs = {}
    if serial_port:
        kwargs["serial_port"] = serial_port
    if mac_address:
        kwargs["mac_address"] = mac_address

    config = BoardConfig.from_name(board_name, **kwargs)

    # Set up callbacks
    def on_chunk(data, elapsed):
        if verbose:
            samples = data["data"].shape[1]
            print(f"  Recording: {elapsed:.1f}s ({samples} samples)", end="\r")

    def on_complete(path):
        if verbose:
            print(f"\n  Saved: {path}")

    # Record
    if verbose:
        print(f"Recording from {config.board_name}...")
        print(f"  Duration: {duration_sec}s")

    recorder = StreamRecorder(config, output_dir=output_dir)
    return recorder.record(
        duration_sec=duration_sec,
        on_chunk=on_chunk,
        on_complete=on_complete,
    )
