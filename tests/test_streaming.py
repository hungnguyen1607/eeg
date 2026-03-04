"""Tests for the streaming module."""

import time
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Check if brainflow is available
try:
    import brainflow
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False


# Skip all tests if brainflow is not installed
pytestmark = pytest.mark.skipif(
    not BRAINFLOW_AVAILABLE,
    reason="brainflow not installed"
)


class TestBoardConfig:
    """Tests for BoardConfig class."""

    def test_synthetic_config(self):
        """Test creating synthetic board config."""
        from eeg_platform.streaming import BoardConfig

        config = BoardConfig.synthetic()

        assert config.board_name == "Synthetic Board"
        assert config.serial_port is None
        assert config.mac_address is None

    def test_cyton_config(self):
        """Test creating Cyton config."""
        from eeg_platform.streaming import BoardConfig

        config = BoardConfig.cyton("COM3")

        assert config.board_name == "OpenBCI Cyton"
        assert config.serial_port == "COM3"

    def test_muse_2_config(self):
        """Test creating Muse 2 config."""
        from eeg_platform.streaming import BoardConfig

        config = BoardConfig.muse_2()
        assert config.board_name == "Muse 2"

        config_with_mac = BoardConfig.muse_2("AA:BB:CC:DD:EE:FF")
        assert config_with_mac.mac_address == "AA:BB:CC:DD:EE:FF"

    def test_from_name(self):
        """Test creating config from board name."""
        from eeg_platform.streaming import BoardConfig

        config = BoardConfig.from_name("synthetic")
        assert config.board_name == "Synthetic Board"

        # Test with different name formats
        config = BoardConfig.from_name("muse-2")
        assert config.board_name == "Muse 2"

        config = BoardConfig.from_name("MUSE_S")
        assert config.board_name == "Muse S"

    def test_from_name_invalid(self):
        """Test that invalid board name raises error."""
        from eeg_platform.streaming import BoardConfig

        with pytest.raises(ValueError, match="Unknown board"):
            BoardConfig.from_name("invalid_board")

    def test_from_name_missing_serial(self):
        """Test that missing serial port raises error for Cyton."""
        from eeg_platform.streaming import BoardConfig

        with pytest.raises(ValueError, match="requires serial_port"):
            BoardConfig.from_name("cyton")


class TestListBoards:
    """Tests for board listing functions."""

    def test_list_boards(self):
        """Test listing available boards."""
        from eeg_platform.streaming import list_boards

        boards = list_boards()

        assert len(boards) > 0
        assert any(b["key"] == "synthetic" for b in boards)

        # Check board structure
        for board in boards:
            assert "key" in board
            assert "name" in board
            assert "channels" in board
            assert "sfreq" in board

    def test_get_board_info(self):
        """Test getting board info."""
        from eeg_platform.streaming import get_board_info

        info = get_board_info("synthetic")

        assert info["name"] == "Synthetic Board"
        assert info["channels"] == 16
        assert info["sfreq"] == 250

    def test_get_board_info_invalid(self):
        """Test that invalid board name raises error."""
        from eeg_platform.streaming import get_board_info

        with pytest.raises(ValueError, match="Unknown board"):
            get_board_info("invalid_board")


class TestBrainFlowClient:
    """Tests for BrainFlowClient class."""

    def test_client_creation(self):
        """Test creating a BrainFlow client."""
        from eeg_platform.streaming import BrainFlowClient, BoardConfig

        config = BoardConfig.synthetic()
        client = BrainFlowClient(config)

        assert client.config == config
        assert not client.is_streaming

    def test_client_properties(self):
        """Test client properties before streaming."""
        from eeg_platform.streaming import BrainFlowClient, BoardConfig

        config = BoardConfig.synthetic()
        client = BrainFlowClient(config)

        assert client.sfreq > 0
        assert client.n_channels > 0
        assert len(client.channel_names) == client.n_channels
        assert len(client.eeg_channels) > 0

    def test_synthetic_streaming(self):
        """Test streaming from synthetic board."""
        from eeg_platform.streaming import BrainFlowClient, BoardConfig

        config = BoardConfig.synthetic()
        client = BrainFlowClient(config)

        try:
            client.start()
            assert client.is_streaming

            # Wait for some data
            time.sleep(0.5)

            # Check data count
            count = client.get_data_count()
            assert count > 0

            # Get data
            data = client.get_data()

            assert "data" in data
            assert "ch_names" in data
            assert "sfreq" in data
            assert "time" in data

            assert data["data"].shape[0] == client.n_channels
            assert data["data"].shape[1] > 0

        finally:
            client.release()
            assert not client.is_streaming

    def test_context_manager(self):
        """Test using client as context manager."""
        from eeg_platform.streaming import BrainFlowClient, BoardConfig

        config = BoardConfig.synthetic()

        with BrainFlowClient(config) as client:
            assert client.is_streaming
            time.sleep(0.2)
            data = client.get_data()
            assert data["data"].shape[1] > 0

        # Client should be released after context
        assert not client.is_streaming

    def test_get_current_data(self):
        """Test getting current data without clearing buffer."""
        from eeg_platform.streaming import BrainFlowClient, BoardConfig

        config = BoardConfig.synthetic()

        with BrainFlowClient(config) as client:
            time.sleep(0.5)

            # Get current data (doesn't clear buffer)
            n_samples = 100
            data1 = client.get_current_board_data(n_samples)

            # Buffer should still have data
            count = client.get_data_count()
            assert count > 0

    def test_quick_record(self):
        """Test quick recording function."""
        from eeg_platform.streaming import quick_record, BoardConfig

        config = BoardConfig.synthetic()
        data = quick_record(config, duration_sec=1.0)

        assert "data" in data
        assert data["data"].shape[1] > 0
        assert data["sfreq"] > 0


class TestStreamRecorder:
    """Tests for StreamRecorder class."""

    def test_recorder_creation(self):
        """Test creating a recorder."""
        from eeg_platform.streaming import StreamRecorder, BoardConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = BoardConfig.synthetic()
            recorder = StreamRecorder(config, output_dir=tmpdir)

            assert not recorder.is_recording
            assert recorder.output_dir == Path(tmpdir)

    def test_record_session(self):
        """Test recording a session."""
        from eeg_platform.streaming import StreamRecorder, BoardConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = BoardConfig.synthetic()
            recorder = StreamRecorder(config, output_dir=tmpdir)

            path = recorder.record(duration_sec=1.0)

            assert path.exists()
            assert path.suffix == ".csv"

            # Check metadata file exists
            meta_path = path.with_suffix(".json")
            assert meta_path.exists()

            # Verify CSV content
            import pandas as pd
            df = pd.read_csv(path)
            assert len(df) > 0
            assert "time" in df.columns

    def test_record_with_callback(self):
        """Test recording with progress callback."""
        from eeg_platform.streaming import StreamRecorder, BoardConfig

        chunks_received = []

        def on_chunk(data, elapsed):
            chunks_received.append((data, elapsed))

        with tempfile.TemporaryDirectory() as tmpdir:
            config = BoardConfig.synthetic()
            recorder = StreamRecorder(config, output_dir=tmpdir)

            recorder.record(
                duration_sec=1.0,
                chunk_sec=0.2,
                on_chunk=on_chunk,
            )

            assert len(chunks_received) > 0


class TestContinuousRecorder:
    """Tests for ContinuousRecorder class."""

    def test_continuous_recorder(self):
        """Test continuous recording with start/stop."""
        from eeg_platform.streaming import ContinuousRecorder, BoardConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = BoardConfig.synthetic()
            recorder = ContinuousRecorder(config, output_dir=tmpdir)

            recorder.start()
            assert recorder.is_recording

            time.sleep(0.5)
            recorder.collect()

            time.sleep(0.5)
            path = recorder.stop()

            assert not recorder.is_recording
            assert path.exists()

    def test_continuous_recorder_context(self):
        """Test continuous recorder as context manager."""
        from eeg_platform.streaming import ContinuousRecorder, BoardConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = BoardConfig.synthetic()

            with ContinuousRecorder(config, output_dir=tmpdir) as recorder:
                assert recorder.is_recording
                time.sleep(0.5)
                recorder.collect()

            assert not recorder.is_recording


class TestRecordSession:
    """Tests for record_session convenience function."""

    def test_record_session_synthetic(self):
        """Test record_session with synthetic board."""
        from eeg_platform.streaming import record_session

        with tempfile.TemporaryDirectory() as tmpdir:
            path = record_session(
                board_name="synthetic",
                duration_sec=1.0,
                output_dir=tmpdir,
                verbose=False,
            )

            assert path.exists()
            assert path.suffix == ".csv"


class TestStreamToCallback:
    """Tests for stream_to_callback function."""

    def test_stream_to_callback(self):
        """Test streaming to callback function."""
        from eeg_platform.streaming import stream_to_callback, BoardConfig

        chunks = []

        def callback(data):
            chunks.append(data)

        config = BoardConfig.synthetic()
        stream_to_callback(
            config,
            callback,
            duration_sec=1.0,
            chunk_sec=0.2,
        )

        assert len(chunks) > 0
        assert all("data" in c for c in chunks)


class TestDataCompatibility:
    """Tests for data format compatibility with rest of platform."""

    def test_data_format_compatible(self):
        """Test that streaming data is compatible with pipeline."""
        from eeg_platform.streaming import quick_record, BoardConfig
        from eeg_platform.preprocess.clean import clean_signal
        from eeg_platform.features.spectral import compute_psd_welch

        config = BoardConfig.synthetic()
        data = quick_record(config, duration_sec=2.0)

        # Should be able to use with clean_signal
        cleaned = clean_signal(
            data["data"],
            data["sfreq"],
            bandpass=(1.0, 45.0),
        )

        assert cleaned.shape == data["data"].shape

        # Should be able to compute PSD
        freqs, psd = compute_psd_welch(
            cleaned,
            data["sfreq"],
            nperseg=min(256, cleaned.shape[1]),
        )

        assert len(freqs) > 0
        assert psd.shape[0] == data["data"].shape[0]
