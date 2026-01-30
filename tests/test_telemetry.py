# Copyright 2024-2026 Hewlett Packard Enterprise Development LP
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for telemetry classes and data structures.
"""
import pytest
from unittest.mock import MagicMock, patch
import time


class TestTelemetryBase:
    """Tests for the TelemetryBase abstract class."""
    
    def test_base_supported_fields(self, th):
        """TelemetryBase should have minimal supported fields."""
        assert "vendor" in th.TelemetryBase.supported
        assert "model" in th.TelemetryBase.supported
        assert "device_id" in th.TelemetryBase.supported
    
    def test_schema_returns_supported(self, th):
        """schema() should return the supported fields list."""
        tel = th.TelemetryBase()
        schema = tel.schema()
        
        assert schema == th.TelemetryBase.supported
    
    def test_read_not_implemented(self, th):
        """read() should raise NotImplementedError in base class."""
        tel = th.TelemetryBase()
        
        with pytest.raises(NotImplementedError):
            tel.read()
    
    def test_get_stats_returns_empty(self, th):
        """get_stats() should return empty dict in base class."""
        tel = th.TelemetryBase()
        stats = tel.get_stats()
        
        assert stats == {}
    
    def test_reset_stats_is_noop(self, th):
        """reset_stats() should be a no-op in base class."""
        tel = th.TelemetryBase()
        # Should not raise
        tel.reset_stats()
    
    def test_shutdown_is_noop(self, th):
        """shutdown() should be a no-op in base class."""
        tel = th.TelemetryBase()
        # Should not raise
        tel.shutdown()


class TestCpuTelemetry:
    """Tests for the CPU fallback telemetry class."""
    
    def test_cpu_telemetry_init(self, th):
        """CpuTelemetry should initialize correctly."""
        tel = th.CpuTelemetry(index=0)
        
        assert tel.idx == 0
        assert hasattr(tel, 'hostname')
        assert hasattr(tel, '_model')
    
    def test_cpu_telemetry_read(self, th):
        """CpuTelemetry.read() should return valid data."""
        tel = th.CpuTelemetry(index=0)
        data = tel.read()
        
        assert "vendor" in data
        assert "model" in data
        assert "device_id" in data
        assert data["device_id"] == 0
        assert "hostname" in data
    
    def test_cpu_telemetry_different_index(self, th):
        """CpuTelemetry should respect device index."""
        tel = th.CpuTelemetry(index=3)
        data = tel.read()
        
        assert data["device_id"] == 3


class TestIntelTelemetry:
    """Tests for the Intel GPU telemetry class (placeholder)."""
    
    def test_intel_telemetry_init(self, th):
        """IntelTelemetry should initialize correctly."""
        tel = th.IntelTelemetry(index=0)
        
        assert tel.idx == 0
        assert hasattr(tel, 'hostname')
    
    def test_intel_telemetry_read(self, th):
        """IntelTelemetry.read() should return valid data."""
        tel = th.IntelTelemetry(index=1)
        data = tel.read()
        
        assert data["vendor"] == "Intel"
        assert data["model"] == "Intel_GPU"
        assert data["device_id"] == 1
        assert "hostname" in data


class TestNVMLTelemetryFields:
    """Tests for NVIDIA telemetry field definitions (no actual GPU needed)."""
    
    def test_nvml_supported_fields(self, th):
        """NVMLTelemetry should define comprehensive supported fields."""
        expected_fields = [
            "hostname", "vendor", "model", "device_id", "serial",
            "sm_util", "mem_bw_util", "mem_util",
            "gpu_clock", "mem_clock",
            "power_W", "temp_gpu_C", "temp_hbm_C",
            "mem_used_MB", "mem_total_MB", "mem_free_MB",
            "hw_slowdown", "sw_slowdown", "power_limit", "throttled",
        ]
        
        for field in expected_fields:
            assert field in th.NVMLTelemetry.supported, f"Missing field: {field}"
    
    def test_nvml_throttle_fields(self, th):
        """NVML telemetry should include throttle detection fields."""
        throttle_fields = ["hw_slowdown", "sw_slowdown", "power_limit", "throttled"]
        
        for field in throttle_fields:
            assert field in th.NVMLTelemetry.supported


class TestRocmTelemetryFields:
    """Tests for AMD ROCm telemetry field definitions (no actual GPU needed)."""
    
    def test_rocm_gpu_fields(self, th):
        """RocmTelemetry should define GPU-specific fields."""
        # Check GPU fields (class-level constant)
        expected_fields = [
            "vendor", "model", "device_id", "serial",
            "gpu_clock", "mem_clock", "power_W",
        ]
        
        for field in expected_fields:
            assert field in th.RocmTelemetry.GPU_FIELDS, f"Missing GPU field: {field}"
    
    def test_rocm_cpu_fields(self, th):
        """RocmTelemetry should define CPU-specific fields."""
        # Check CPU fields (for APU mode)
        expected_fields = [
            "vendor", "model", "device_id", "serial",
        ]
        
        for field in expected_fields:
            assert field in th.RocmTelemetry.CPU_FIELDS, f"Missing CPU field: {field}"


class TestTelemetryThread:
    """Tests for the background telemetry thread."""
    
    def test_thread_initialization(self, th):
        """TelemetryThread should initialize correctly."""
        mock_tel = MagicMock()
        mock_tel.read.return_value = {"vendor": "TEST", "device_id": 0}
        
        import torch
        device = torch.device("cpu")
        
        thread = th.TelemetryThread(mock_tel, device, sample_interval_ms=100)
        
        assert thread.sample_interval_ms == 100
        assert thread.running is False
        assert thread.active is False
        assert thread.latest_reading == {}
    
    def test_thread_start_stop(self, th):
        """TelemetryThread should start and stop cleanly."""
        mock_tel = MagicMock()
        mock_tel.read.return_value = {"vendor": "TEST", "device_id": 0}
        
        import torch
        device = torch.device("cpu")
        
        thread = th.TelemetryThread(mock_tel, device, sample_interval_ms=50)
        
        thread.start()
        assert thread.running is True
        assert thread.thread is not None
        
        time.sleep(0.1)  # Let thread run briefly
        
        thread.stop()
        assert thread.running is False
    
    def test_get_latest_returns_data(self, th):
        """get_latest() should return telemetry data."""
        mock_tel = MagicMock()
        mock_tel.read.return_value = {"vendor": "TEST", "model": "TestGPU", "device_id": 0}
        
        import torch
        device = torch.device("cpu")
        
        thread = th.TelemetryThread(mock_tel, device, sample_interval_ms=20)
        thread.start()
        
        time.sleep(0.1)  # Let thread collect some data
        
        latest = thread.get_latest()
        
        thread.stop()
        
        assert "vendor" in latest
        assert latest["vendor"] == "TEST"
    
    def test_set_active_flag(self, th):
        """set_active() should update the active flag."""
        mock_tel = MagicMock()
        mock_tel.read.return_value = {"device_id": 0}
        
        import torch
        device = torch.device("cpu")
        
        thread = th.TelemetryThread(mock_tel, device)
        
        assert thread.active is False
        
        thread.set_active(True)
        assert thread.active is True
        
        thread.set_active(False)
        assert thread.active is False
    
    def test_iteration_marking(self, th):
        """mark_iteration_start/end should track iterations."""
        mock_tel = MagicMock()
        mock_tel.read.return_value = {"device_id": 0, "power_W": 100}
        
        import torch
        device = torch.device("cpu")
        
        thread = th.TelemetryThread(mock_tel, device, sample_interval_ms=10)
        thread.start()
        
        thread.mark_iteration_start(0)
        time.sleep(0.05)
        thread.mark_iteration_end(0)
        
        thread.mark_iteration_start(1)
        time.sleep(0.05)
        thread.mark_iteration_end(1)
        
        time.sleep(0.05)  # Let thread collect
        
        thread.stop()
        
        # Should have iteration data
        assert 0 in thread.iteration_samples or 1 in thread.iteration_samples
    
    def test_get_iteration_telemetry_aggregates(self, th):
        """get_iteration_telemetry() should aggregate samples."""
        mock_tel = MagicMock()
        mock_tel.read.return_value = {"device_id": 0, "power_W": 100, "temp_gpu_C": 50}
        
        import torch
        device = torch.device("cpu")
        
        thread = th.TelemetryThread(mock_tel, device, sample_interval_ms=10)
        thread.start()
        
        thread.mark_iteration_start(5)
        time.sleep(0.05)
        thread.mark_iteration_end(5)
        
        time.sleep(0.02)  # Let thread finish
        
        data = thread.get_iteration_telemetry(5)
        
        thread.stop()
        
        # Should return data (either collected or latest)
        assert data is not None
        assert "device_id" in data


class TestMakeTelemetry:
    """Tests for the telemetry factory function."""
    
    def test_cpu_fallback(self, th):
        """make_telemetry should fall back to CPU when no GPU tools available."""
        import torch
        device = torch.device("cpu")
        
        # Mock shutil.which to return None (no nvidia-smi or rocm-smi)
        with patch('shutil.which', return_value=None):
            tel = th.make_telemetry(0, device)
        
        # Should get CpuTelemetry
        assert isinstance(tel, th.CpuTelemetry)
    
    def test_xpu_returns_intel(self, th):
        """make_telemetry should return IntelTelemetry for XPU devices."""
        import torch
        
        # Create a mock device that identifies as XPU
        mock_device = MagicMock()
        mock_device.type = "xpu"
        
        with patch('shutil.which', return_value=None):
            tel = th.make_telemetry(0, mock_device)
        
        assert isinstance(tel, th.IntelTelemetry)


class TestTelemetryStatAccumulation:
    """Tests for telemetry statistics accumulation."""
    
    def test_cpu_telemetry_no_stats(self, th):
        """CpuTelemetry should have no stats (expected for CPU)."""
        tel = th.CpuTelemetry(0)
        
        # Read a few times
        for _ in range(5):
            tel.read()
        
        stats = tel.get_stats()
        assert stats == {}  # CPU telemetry has no stats tracking
    
    def test_reset_stats_clears_data(self, th):
        """reset_stats should work without error."""
        tel = th.CpuTelemetry(0)
        
        # Should not raise
        tel.reset_stats()
