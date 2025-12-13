"""
TensorRT Optimization Module for Real-Time Inference.

Provides tools for:
- Exporting PyTorch models to ONNX
- Converting ONNX to TensorRT engines
- Optimized inference with TensorRT
- INT8 calibration for maximum performance

Usage:
    from src.inference import TensorRTOptimizer
    
    # Export and optimize model
    optimizer = TensorRTOptimizer(model, input_shape=(1, 8, 2))
    optimizer.export_onnx('model.onnx')
    optimizer.build_engine('model.trt', fp16=True)
    
    # Run inference
    output = optimizer.inference(input_data)

Requirements:
    - tensorrt >= 8.0
    - torch >= 1.9
    - onnx >= 1.10
    - pycuda (for TensorRT inference)

References:
- TensorRT Developer Guide: https://docs.nvidia.com/deeplearning/tensorrt/
- PyTorch ONNX Export: https://pytorch.org/docs/stable/onnx.html
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TensorRTConfig:
    """TensorRT optimization configuration."""
    # Precision modes
    fp32: bool = True
    fp16: bool = False
    int8: bool = False
    
    # Optimization settings
    max_workspace_size: int = 1 << 30  # 1GB
    max_batch_size: int = 1
    min_batch_size: int = 1
    opt_batch_size: int = 1
    
    # Dynamic shapes
    use_dynamic_shapes: bool = False
    min_shapes: Optional[Dict[str, Tuple]] = None
    opt_shapes: Optional[Dict[str, Tuple]] = None
    max_shapes: Optional[Dict[str, Tuple]] = None
    
    # INT8 calibration
    calibration_images: int = 500
    calibration_batch_size: int = 8
    
    # Engine settings
    dla_core: int = -1  # -1 for GPU, 0/1 for DLA cores (Jetson)
    strict_types: bool = False


# =============================================================================
# ONNX Export
# =============================================================================

class ONNXExporter:
    """Export PyTorch models to ONNX format."""
    
    def __init__(
        self,
        model,
        input_names: List[str] = None,
        output_names: List[str] = None,
        dynamic_axes: Dict[str, Dict[int, str]] = None,
        opset_version: int = 14,
    ):
        """
        Initialize ONNX exporter.
        
        Args:
            model: PyTorch model
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes specification
            opset_version: ONNX opset version
        """
        self.model = model
        self.input_names = input_names or ['input']
        self.output_names = output_names or ['output']
        self.dynamic_axes = dynamic_axes
        self.opset_version = opset_version
    
    def export(
        self,
        output_path: str,
        input_shapes: Dict[str, Tuple],
        device: str = 'cuda',
    ) -> str:
        """
        Export model to ONNX.
        
        Args:
            output_path: Output ONNX file path
            input_shapes: Dictionary of input shapes
            device: Export device
        
        Returns:
            Path to exported ONNX file
        """
        import torch
        
        self.model.eval()
        self.model.to(device)
        
        # Create dummy inputs
        dummy_inputs = {}
        for name, shape in input_shapes.items():
            dummy_inputs[name] = torch.randn(*shape, device=device)
        
        # Handle single or multiple inputs
        if len(dummy_inputs) == 1:
            dummy_input = list(dummy_inputs.values())[0]
        else:
            dummy_input = tuple(dummy_inputs.values())
        
        # Export
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                input_names=self.input_names,
                output_names=self.output_names,
                dynamic_axes=self.dynamic_axes,
                opset_version=self.opset_version,
                do_constant_folding=True,
            )
        
        logger.info(f"ONNX model exported to {output_path}")
        
        # Verify ONNX model
        self._verify_onnx(output_path)
        
        return output_path
    
    def _verify_onnx(self, onnx_path: str):
        """Verify ONNX model is valid."""
        try:
            import onnx
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            logger.info("ONNX model verification passed")
        except Exception as e:
            logger.warning(f"ONNX verification failed: {e}")
    
    def simplify(self, onnx_path: str, output_path: str = None) -> str:
        """
        Simplify ONNX model using onnx-simplifier.
        
        Args:
            onnx_path: Input ONNX file
            output_path: Output simplified ONNX file
        
        Returns:
            Path to simplified model
        """
        try:
            import onnx
            from onnxsim import simplify
            
            model = onnx.load(onnx_path)
            model_simplified, check = simplify(model)
            
            if not check:
                logger.warning("ONNX simplification check failed")
            
            output_path = output_path or onnx_path.replace('.onnx', '_simplified.onnx')
            onnx.save(model_simplified, output_path)
            
            logger.info(f"Simplified ONNX saved to {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("onnx-simplifier not installed. Run: pip install onnx-simplifier")
            return onnx_path


# =============================================================================
# INT8 Calibrator
# =============================================================================

class INT8Calibrator:
    """INT8 calibration for TensorRT."""
    
    def __init__(
        self,
        calibration_data,
        cache_file: str = 'calibration.cache',
        batch_size: int = 8,
    ):
        """
        Initialize INT8 calibrator.
        
        Args:
            calibration_data: Data loader or array for calibration
            cache_file: Path to calibration cache
            batch_size: Calibration batch size
        """
        self.calibration_data = calibration_data
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_batch = 0
        self.max_batches = len(calibration_data) if hasattr(calibration_data, '__len__') else 100
    
    def get_batch_size(self):
        """Return calibration batch size."""
        return self.batch_size
    
    def get_batch(self, names):
        """Get next calibration batch."""
        if self.current_batch >= self.max_batches:
            return None
        
        try:
            if hasattr(self.calibration_data, '__iter__'):
                batch = next(iter(self.calibration_data))
            else:
                start = self.current_batch * self.batch_size
                end = start + self.batch_size
                batch = self.calibration_data[start:end]
            
            self.current_batch += 1
            
            if isinstance(batch, dict):
                batch = batch[names[0]] if names else list(batch.values())[0]
            
            return [batch.numpy() if hasattr(batch, 'numpy') else batch]
            
        except StopIteration:
            return None
    
    def read_calibration_cache(self):
        """Read calibration cache."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        """Write calibration cache."""
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


# =============================================================================
# TensorRT Engine Builder
# =============================================================================

class TensorRTBuilder:
    """Build TensorRT engines from ONNX models."""
    
    def __init__(self, config: TensorRTConfig = None):
        """
        Initialize TensorRT builder.
        
        Args:
            config: TensorRT configuration
        """
        self.config = config or TensorRTConfig()
        self._check_tensorrt()
    
    def _check_tensorrt(self):
        """Check TensorRT availability."""
        try:
            import tensorrt as trt
            self.trt = trt
            self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            logger.info(f"TensorRT version: {trt.__version__}")
        except ImportError:
            raise ImportError(
                "TensorRT not installed. Install with: "
                "pip install tensorrt (CUDA toolkit required)"
            )
    
    def build_engine(
        self,
        onnx_path: str,
        engine_path: str,
        calibrator=None,
    ) -> str:
        """
        Build TensorRT engine from ONNX.
        
        Args:
            onnx_path: Input ONNX file path
            engine_path: Output TensorRT engine path
            calibrator: INT8 calibrator (optional)
        
        Returns:
            Path to TensorRT engine
        """
        trt = self.trt
        
        # Create builder
        builder = trt.Builder(self.TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.TRT_LOGGER)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f"ONNX parse error: {parser.get_error(i)}")
                raise RuntimeError("Failed to parse ONNX model")
        
        logger.info(f"ONNX model parsed: {network.num_inputs} inputs, {network.num_outputs} outputs")
        
        # Create config
        config = builder.create_builder_config()
        config.max_workspace_size = self.config.max_workspace_size
        
        # Set precision
        if self.config.fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 mode enabled")
        
        if self.config.int8 and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            if calibrator:
                config.int8_calibrator = calibrator
            logger.info("INT8 mode enabled")
        
        # Set dynamic shapes if enabled
        if self.config.use_dynamic_shapes:
            self._set_dynamic_shapes(builder, network, config)
        
        # Build engine
        logger.info("Building TensorRT engine (this may take a while)...")
        start_time = time.time()
        
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        build_time = time.time() - start_time
        logger.info(f"Engine built in {build_time:.1f}s")
        
        # Save engine
        os.makedirs(os.path.dirname(engine_path) or '.', exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        logger.info(f"TensorRT engine saved to {engine_path}")
        
        return engine_path
    
    def _set_dynamic_shapes(self, builder, network, config):
        """Configure dynamic shapes."""
        profile = builder.create_optimization_profile()
        
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            name = input_tensor.name
            
            min_shape = self.config.min_shapes.get(name, (1,) * len(input_tensor.shape))
            opt_shape = self.config.opt_shapes.get(name, min_shape)
            max_shape = self.config.max_shapes.get(name, opt_shape)
            
            profile.set_shape(name, min_shape, opt_shape, max_shape)
        
        config.add_optimization_profile(profile)


# =============================================================================
# TensorRT Runtime
# =============================================================================

class TensorRTRuntime:
    """TensorRT engine inference runtime."""
    
    def __init__(self, engine_path: str, device_id: int = 0):
        """
        Initialize TensorRT runtime.
        
        Args:
            engine_path: Path to TensorRT engine
            device_id: CUDA device ID
        """
        self.engine_path = engine_path
        self.device_id = device_id
        
        self._init_tensorrt()
        self._load_engine()
        self._allocate_buffers()
    
    def _init_tensorrt(self):
        """Initialize TensorRT and CUDA."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            self.trt = trt
            self.cuda = cuda
            self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
        except ImportError as e:
            raise ImportError(
                f"TensorRT runtime dependencies not installed: {e}\n"
                "Install with: pip install tensorrt pycuda"
            )
    
    def _load_engine(self):
        """Load TensorRT engine."""
        trt = self.trt
        
        runtime = trt.Runtime(self.TRT_LOGGER)
        
        with open(self.engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine from {self.engine_path}")
        
        self.context = self.engine.create_execution_context()
        
        logger.info(f"TensorRT engine loaded: {self.engine_path}")
    
    def _allocate_buffers(self):
        """Allocate GPU buffers."""
        cuda = self.cuda
        
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for i in range(self.engine.num_bindings):
            binding = self.engine.get_binding_name(i)
            dtype = self.trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)
            
            # Handle dynamic shapes
            if -1 in shape:
                shape = tuple(abs(s) for s in shape)
            
            size = np.prod(shape)
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(int(size), dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(i):
                self.inputs.append({
                    'name': binding,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': dtype,
                })
            else:
                self.outputs.append({
                    'name': binding,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': dtype,
                })
        
        logger.info(f"Allocated {len(self.inputs)} input(s), {len(self.outputs)} output(s)")
    
    def inference(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run inference.
        
        Args:
            inputs: Input array or dictionary of inputs
        
        Returns:
            Output array or dictionary of outputs
        """
        cuda = self.cuda
        
        # Handle single input
        if isinstance(inputs, np.ndarray):
            inputs = {self.inputs[0]['name']: inputs}
        
        # Copy inputs to device
        for inp in self.inputs:
            data = inputs.get(inp['name'])
            if data is not None:
                np.copyto(inp['host'], data.ravel())
                cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        
        # Execute
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )
        
        # Copy outputs from device
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        
        # Synchronize
        self.stream.synchronize()
        
        # Return outputs
        if len(self.outputs) == 1:
            return self.outputs[0]['host'].reshape(self.outputs[0]['shape'])
        
        return {
            out['name']: out['host'].reshape(out['shape'])
            for out in self.outputs
        }
    
    def benchmark(
        self,
        input_shape: Tuple,
        num_iterations: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            input_shape: Input tensor shape
            num_iterations: Number of iterations
            warmup: Warmup iterations
        
        Returns:
            Performance metrics
        """
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(warmup):
            self.inference(dummy_input)
        
        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.inference(dummy_input)
            latencies.append(time.perf_counter() - start)
        
        latencies = np.array(latencies) * 1000  # Convert to ms
        
        return {
            'latency_mean_ms': latencies.mean(),
            'latency_std_ms': latencies.std(),
            'latency_min_ms': latencies.min(),
            'latency_max_ms': latencies.max(),
            'latency_p95_ms': np.percentile(latencies, 95),
            'latency_p99_ms': np.percentile(latencies, 99),
            'throughput_fps': 1000.0 / latencies.mean(),
        }
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'context') and self.context:
            del self.context
        if hasattr(self, 'engine') and self.engine:
            del self.engine


# =============================================================================
# High-Level Optimizer
# =============================================================================

class TensorRTOptimizer:
    """
    High-level TensorRT optimization pipeline.
    
    Provides end-to-end workflow:
    1. Export PyTorch model to ONNX
    2. Optionally simplify ONNX
    3. Build TensorRT engine
    4. Run optimized inference
    
    Usage:
        optimizer = TensorRTOptimizer(model)
        optimizer.optimize('model.trt', input_shape=(1, 8, 2), fp16=True)
        output = optimizer.inference(input_data)
    """
    
    def __init__(
        self,
        model,
        input_names: List[str] = None,
        output_names: List[str] = None,
        dynamic_axes: Dict = None,
    ):
        """
        Initialize optimizer.
        
        Args:
            model: PyTorch model to optimize
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes specification
        """
        self.model = model
        self.input_names = input_names or ['input']
        self.output_names = output_names or ['output']
        self.dynamic_axes = dynamic_axes
        
        self.onnx_path = None
        self.engine_path = None
        self.runtime = None
    
    def export_onnx(
        self,
        output_path: str,
        input_shapes: Dict[str, Tuple],
        simplify: bool = True,
    ) -> str:
        """
        Export model to ONNX.
        
        Args:
            output_path: Output ONNX path
            input_shapes: Input shapes dictionary
            simplify: Whether to simplify ONNX
        
        Returns:
            Path to ONNX file
        """
        exporter = ONNXExporter(
            self.model,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
        )
        
        self.onnx_path = exporter.export(output_path, input_shapes)
        
        if simplify:
            self.onnx_path = exporter.simplify(self.onnx_path)
        
        return self.onnx_path
    
    def build_engine(
        self,
        output_path: str,
        fp16: bool = True,
        int8: bool = False,
        calibration_data=None,
        max_workspace_size: int = 1 << 30,
    ) -> str:
        """
        Build TensorRT engine.
        
        Args:
            output_path: Output engine path
            fp16: Enable FP16 precision
            int8: Enable INT8 precision
            calibration_data: Data for INT8 calibration
            max_workspace_size: Maximum workspace size
        
        Returns:
            Path to TensorRT engine
        """
        if self.onnx_path is None:
            raise RuntimeError("Export ONNX first using export_onnx()")
        
        config = TensorRTConfig(
            fp16=fp16,
            int8=int8,
            max_workspace_size=max_workspace_size,
        )
        
        builder = TensorRTBuilder(config)
        
        calibrator = None
        if int8 and calibration_data is not None:
            calibrator = INT8Calibrator(calibration_data)
        
        self.engine_path = builder.build_engine(
            self.onnx_path,
            output_path,
            calibrator=calibrator,
        )
        
        return self.engine_path
    
    def optimize(
        self,
        output_path: str,
        input_shapes: Dict[str, Tuple],
        fp16: bool = True,
        int8: bool = False,
        calibration_data=None,
        simplify_onnx: bool = True,
    ) -> str:
        """
        Full optimization pipeline.
        
        Args:
            output_path: Output TensorRT engine path
            input_shapes: Input shapes dictionary
            fp16: Enable FP16
            int8: Enable INT8
            calibration_data: INT8 calibration data
            simplify_onnx: Simplify ONNX model
        
        Returns:
            Path to optimized TensorRT engine
        """
        # Create ONNX path
        onnx_path = output_path.replace('.trt', '.onnx').replace('.engine', '.onnx')
        if onnx_path == output_path:
            onnx_path = output_path + '.onnx'
        
        # Export and build
        self.export_onnx(onnx_path, input_shapes, simplify=simplify_onnx)
        self.build_engine(output_path, fp16=fp16, int8=int8, calibration_data=calibration_data)
        
        # Load runtime
        self.load_engine(output_path)
        
        return self.engine_path
    
    def load_engine(self, engine_path: str):
        """Load TensorRT engine for inference."""
        self.engine_path = engine_path
        self.runtime = TensorRTRuntime(engine_path)
    
    def inference(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Run inference with TensorRT engine."""
        if self.runtime is None:
            raise RuntimeError("Load engine first using load_engine()")
        
        return self.runtime.inference(inputs)
    
    def benchmark(
        self,
        input_shape: Tuple,
        num_iterations: int = 100,
    ) -> Dict[str, float]:
        """Benchmark TensorRT inference performance."""
        if self.runtime is None:
            raise RuntimeError("Load engine first using load_engine()")
        
        return self.runtime.benchmark(input_shape, num_iterations)


# =============================================================================
# Utility Functions
# =============================================================================

def check_tensorrt_available() -> bool:
    """Check if TensorRT is available."""
    try:
        import tensorrt
        return True
    except ImportError:
        return False


def get_tensorrt_version() -> Optional[str]:
    """Get TensorRT version."""
    try:
        import tensorrt as trt
        return trt.__version__
    except ImportError:
        return None


def compare_pytorch_tensorrt(
    pytorch_model,
    tensorrt_optimizer: TensorRTOptimizer,
    test_input: np.ndarray,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> Dict[str, Any]:
    """
    Compare PyTorch and TensorRT outputs.
    
    Args:
        pytorch_model: PyTorch model
        tensorrt_optimizer: Optimized TensorRT model
        test_input: Test input array
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        Comparison results
    """
    import torch
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        torch_input = torch.from_numpy(test_input).float()
        pytorch_output = pytorch_model(torch_input).numpy()
    
    # TensorRT inference
    tensorrt_output = tensorrt_optimizer.inference(test_input.astype(np.float32))
    
    # Compare
    is_close = np.allclose(pytorch_output, tensorrt_output, rtol=rtol, atol=atol)
    max_diff = np.abs(pytorch_output - tensorrt_output).max()
    mean_diff = np.abs(pytorch_output - tensorrt_output).mean()
    
    return {
        'match': is_close,
        'max_diff': float(max_diff),
        'mean_diff': float(mean_diff),
        'pytorch_output_shape': pytorch_output.shape,
        'tensorrt_output_shape': tensorrt_output.shape,
    }


def optimize_for_jetson(
    model,
    output_path: str,
    input_shapes: Dict[str, Tuple],
    dla_core: int = 0,
) -> str:
    """
    Optimize model for NVIDIA Jetson platform.
    
    Args:
        model: PyTorch model
        output_path: Output engine path
        input_shapes: Input shapes
        dla_core: DLA core to use (0 or 1)
    
    Returns:
        Path to optimized engine
    """
    config = TensorRTConfig(
        fp16=True,
        dla_core=dla_core,
        max_workspace_size=1 << 28,  # 256MB for Jetson
    )
    
    optimizer = TensorRTOptimizer(model)
    
    # Export ONNX
    onnx_path = output_path.replace('.trt', '.onnx')
    optimizer.export_onnx(onnx_path, input_shapes)
    
    # Build with DLA
    builder = TensorRTBuilder(config)
    engine_path = builder.build_engine(onnx_path, output_path)
    
    return engine_path
