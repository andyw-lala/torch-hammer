# Copyright 2024-2026 Hewlett Packard Enterprise Development LP
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for utility functions: Timer, VerbosePrinter, helper functions.
"""
import time
import logging
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestGflopsCalculation:
    """Tests for GFLOP/s calculation helper."""
    
    def test_simple_calculation(self, th):
        """Basic GFLOP/s calculation should work correctly."""
        # 1e12 flops in 1 second = 1000 GFLOP/s
        result = th.gflops(1e12, 1.0)
        assert result == 1000.0
    
    def test_scaled_calculation(self, th):
        """Scaled values should calculate correctly."""
        # 2e12 flops in 2 seconds = 1000 GFLOP/s
        result = th.gflops(2e12, 2.0)
        assert result == 1000.0
    
    def test_small_time(self, th):
        """Small time values should produce high GFLOP/s."""
        # 1e12 flops in 0.1 seconds = 10000 GFLOP/s
        result = th.gflops(1e12, 0.1)
        assert abs(result - 10000.0) < 0.01
    
    def test_zero_time_raises(self, th):
        """Zero time should raise ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            th.gflops(1e12, 0.0)


class TestTimer:
    """Tests for the Timer context manager."""
    
    def test_cpu_timing(self, th):
        """Timer should measure CPU time correctly."""
        device = torch.device("cpu")
        
        with th.Timer(device) as t:
            time.sleep(0.1)  # Sleep 100ms
        # Allow generous tolerance for CI runners (sleep(0.1) + overhead)
        assert timer.elapsed >= 0.09  # At least ~90ms
        assert timer.elapsed < 0.3    # But less than 300ms (CI can be slow)
 
    def test_cpu_timer_attributes(self, th):
        """CPU Timer should have correct attributes."""
        device = torch.device("cpu")
        timer = th.Timer(device)
        
        assert timer.cuda is False
    
    def test_cuda_timer_attributes(self, th):
        """CUDA Timer should have correct attributes when CUDA available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device("cuda:0")
        timer = th.Timer(device)
        
        assert timer.cuda is True
        assert hasattr(timer, 's')
        assert hasattr(timer, 'e')
    
    def test_context_manager_protocol(self, th):
        """Timer should implement context manager protocol."""
        device = torch.device("cpu")
        timer = th.Timer(device)
        
        # Should have __enter__ and __exit__
        assert hasattr(timer, '__enter__')
        assert hasattr(timer, '__exit__')
    
    def test_elapsed_only_available_after_exit(self, th):
        """elapsed attribute should only be set after context exit."""
        device = torch.device("cpu")
        timer = th.Timer(device)
        
        # Before entering, no elapsed
        assert not hasattr(timer, 'elapsed') or timer.elapsed is None
        
        timer.__enter__()
        # During, might not have elapsed yet
        time.sleep(0.01)
        timer.__exit__(None, None, None)
        
        # After exit, should have elapsed
        assert hasattr(timer, 'elapsed')
        assert timer.elapsed > 0


class TestVerbosePrinter:
    """Tests for the VerbosePrinter class."""
    
    def test_initialization(self, th):
        """VerbosePrinter should initialize correctly."""
        logger = logging.getLogger("test")
        schema = ["vendor", "model", "device_id", "sm_util", "power_W"]
        
        printer = th.VerbosePrinter(logger, schema, gpu_id=0)
        
        assert printer.log == logger
        assert printer.gpu_id == 0
        assert "vendor" in printer.schema
        assert "model" in printer.schema
    
    def test_schema_reordering(self, th):
        """VerbosePrinter should put vendor/model first in schema."""
        logger = logging.getLogger("test")
        schema = ["sm_util", "power_W", "vendor", "model", "device_id"]
        
        printer = th.VerbosePrinter(logger, schema, gpu_id=0)
        
        # vendor and model should be first
        assert printer.schema[0] == "vendor"
        assert printer.schema[1] == "model"
    
    def test_emit_logs_header_first_time(self, th):
        """emit() should log header on first call."""
        mock_logger = MagicMock()
        schema = ["vendor", "model", "sm_util"]
        
        printer = th.VerbosePrinter(mock_logger, schema, gpu_id=0)
        
        tel_data = {"vendor": "TEST", "model": "TestGPU", "sm_util": 95}
        printer.emit(0, "gemm", "float32", "gflops", 1000.0, tel_data)
        
        # Should have called info twice (header + data)
        assert mock_logger.info.call_count == 2
    
    def test_emit_includes_all_fields(self, th):
        """emit() should include all schema fields in output."""
        mock_logger = MagicMock()
        schema = ["vendor", "model", "sm_util", "power_W"]
        
        printer = th.VerbosePrinter(mock_logger, schema, gpu_id=0)
        
        tel_data = {"vendor": "NVIDIA", "model": "A100", "sm_util": 95, "power_W": 300.5}
        printer.emit(0, "gemm", "float32", "gflops", 1000.0, tel_data)
        
        # Get the data line (second call)
        data_call = mock_logger.info.call_args_list[1]
        data_line = data_call[0][0]
        
        assert "NVIDIA" in data_line
        assert "A100" in data_line
        assert "95" in data_line


class TestFormatTelemetryCompact:
    """Tests for telemetry formatting helper."""
    
    def test_basic_formatting(self, th):
        """Basic telemetry data should format correctly."""
        tel_data = {
            "device_id": 0,
            "vendor": "NVIDIA",  # Indicates GPU device
            "sm_util": 95,
            "mem_bw_util": 80,
            "temp_gpu_C": 65,
            "power_W": 250.5,
            "gpu_clock": 1500,
        }
        
        result = th._format_telemetry_compact(tel_data)
        
        assert "GPU0" in result
        assert "SM:95%" in result
        assert "Temp:65°C" in result
    
    def test_handles_missing_fields(self, th):
        """Formatter should handle missing fields gracefully."""
        tel_data = {"device_id": 1, "vendor": "NVIDIA"}
        
        # Should not raise
        result = th._format_telemetry_compact(tel_data)
        
        assert "GPU1" in result
    
    def test_cpu_device_formatting(self, th):
        """CPU-only telemetry should not say GPU."""
        tel_data = {
            "device_id": 0,
            "vendor": "CPU",
            "model": "x86_64",
        }
        
        result = th._format_telemetry_compact(tel_data)
        
        # Should show model name, not GPU0
        assert "x86_64" in result
        assert "GPU" not in result
    
    def test_memory_formatting(self, th):
        """Memory usage should format with GB and percentage."""
        tel_data = {
            "device_id": 0,
            "mem_used_MB": 20480,  # 20 GB
            "mem_total_MB": 40960,  # 40 GB
        }
        
        result = th._format_telemetry_compact(tel_data)
        
        assert "Mem:" in result
        assert "GB" in result


class TestValidatePerformance:
    """Tests for hardware performance validation."""
    
    def test_no_baselines_returns_valid(self, th):
        """With no baselines, validation should return valid."""
        result = th.validate_performance(
            model_name="Unknown GPU",
            benchmark="batched_gemm",
            dtype="float32",
            measured_value=50000.0,  # 50 TFLOP/s
            baselines={}
        )
        
        assert result['valid'] is True
        assert result['baseline'] is None
    
    def test_with_target_baselines(self, th):
        """Target-based baselines should validate correctly."""
        baselines = {
            "TestGPU": {
                "benchmarks": {
                    "batched_gemm": {
                        "float32": {
                            "target_gflops": 40000.0,
                            "min_efficiency": 80.0
                        }
                    }
                }
            }
        }
        
        result = th.validate_performance(
            model_name="TestGPU 80GB",
            benchmark="batched_gemm",
            dtype="float32",
            measured_value=35000.0,  # 35 TFLOP/s = 87.5% of target
            unit="gflop/s",
            baselines=baselines
        )
        
        assert result['valid'] is True
        assert result['efficiency'] is not None
        assert 85 < result['efficiency'] < 90
    
    def test_low_efficiency_triggers_warning(self, th):
        """Low efficiency should trigger warning."""
        baselines = {
            "TestGPU": {
                "benchmarks": {
                    "batched_gemm": {
                        "float32": {
                            "target_gflops": 40000.0,
                            "min_efficiency": 80.0
                        }
                    }
                }
            }
        }
        
        result = th.validate_performance(
            model_name="TestGPU 80GB",
            benchmark="batched_gemm",
            dtype="float32",
            measured_value=20000.0,  # 20 TFLOP/s = 50% of target
            unit="gflop/s",
            baselines=baselines
        )
        
        assert result['valid'] is False
        assert result['warning'] is not None
        assert "below target" in result['warning'].lower()


class TestLoadHardwareBaselines:
    """Tests for baseline file loading."""
    
    def test_load_nonexistent_file(self, th):
        """Loading nonexistent file should return empty dict."""
        result = th.load_hardware_baselines("/nonexistent/path/file.json")
        assert result == {}
    
    def test_load_unknown_format(self, th):
        """Unknown file format should return empty dict."""
        result = th.load_hardware_baselines("/some/path/file.txt")
        assert result == {}
