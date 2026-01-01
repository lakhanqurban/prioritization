import numpy as np
import time
import logging
from typing import List, Dict, Any
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

logger = logging.getLogger(__name__)

# GPU acceleration imports (optional)
# CUDA GPU support (NVIDIA) - Only available on systems with NVIDIA GPUs
CUDA_GPU_AVAILABLE = False
cp = None  # type: ignore
CumlAgglomerativeClustering = None  # type: ignore

try:
    import cupy as cp  # type: ignore  # pylint: disable=import-error
    import cupyx  # type: ignore  # pylint: disable=import-error
    # logger.info("🔧 CuPy imported successfully")
    
    # Try to import cuML only if CuPy is available
    try:
        from cuml.cluster import AgglomerativeClustering as CumlAgglomerativeClustering  # type: ignore  # pylint: disable=import-error
        CUDA_GPU_AVAILABLE = True
        # logger.info("🚀 CUDA GPU acceleration available with CuPy and cuML")
    except (ImportError, ModuleNotFoundError):
        # logger.info("📦 cuML not available, CUDA GPU clustering disabled")
        pass
        
except ImportError:
    pass
    # logger.info("📦 CuPy not available (expected on Apple Silicon)")
    # Keep default values: CUDA_GPU_AVAILABLE = False, cp = None, CumlAgglomerativeClustering = None

# Apple Silicon GPU support (Metal Performance Shaders)
try:
    import torch
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        APPLE_GPU_AVAILABLE = True
        # logger.info("🍎 Apple Silicon GPU (Metal Performance Shaders) detected")
    else:
        APPLE_GPU_AVAILABLE = False
        # logger.info("💻 Apple Silicon GPU not available")
except ImportError:
    APPLE_GPU_AVAILABLE = False
    torch = None  # Set to None for safety
    pass
    # logger.info("💻 PyTorch not available")

# Set overall GPU availability
GPU_AVAILABLE = CUDA_GPU_AVAILABLE or APPLE_GPU_AVAILABLE
if not GPU_AVAILABLE:
    pass
    # logger.info("💻 Using CPU-only computation (install PyTorch for Apple Silicon or cuML for NVIDIA GPU acceleration)")

class GPUOperationsMixin:
    """Mixin class for GPU acceleration operations"""
    
    def _init_gpu_operations(self, use_gpu: bool):
        """Initialize GPU operations"""
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            if CUDA_GPU_AVAILABLE:
                logger.info("🚀 Enabling CUDA GPU acceleration")
                self.gpu_type = "CUDA"
            elif APPLE_GPU_AVAILABLE:
                logger.info("🍎 Enabling Apple Silicon GPU acceleration")
                self.gpu_type = "MPS"
        else:
            logger.info("💻 Using CPU computation")
            self.gpu_type = "CPU"

    def _to_gpu(self, data):
        """Transfer data to GPU if available"""
        if not self.use_gpu:
            return data
        
        if self.gpu_type == "CUDA" and CUDA_GPU_AVAILABLE:
            if isinstance(data, np.ndarray):
                return cp.asarray(data)
            return data
        elif self.gpu_type == "MPS" and APPLE_GPU_AVAILABLE:
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data).to('mps')
            return data
        return data
    
    def _to_cpu(self, data):
        """Transfer data back to CPU"""
        if self.gpu_type == "CUDA" and hasattr(data, 'get'):
            return data.get()  # CuPy to NumPy
        elif self.gpu_type == "MPS" and hasattr(data, 'cpu'):
            return data.cpu().numpy()  # PyTorch to NumPy
        return data
    
    def _gpu_dtw_batch(self, curvatures_batch: List[np.ndarray]) -> np.ndarray:
        """
        Accelerated DTW computation for batch of curvature profiles
        """
        if not self.use_gpu or len(curvatures_batch) < 25:
            # Fall back to CPU for small batches (lowered threshold for better GPU utilization)
            return self._cpu_dtw_batch(curvatures_batch)
        
        start_time = time.time()
        
        if self.gpu_type == "CUDA" and CUDA_GPU_AVAILABLE:
            # Use CuPy for CUDA acceleration
            return self._cuda_dtw_batch(curvatures_batch)
        elif self.gpu_type == "MPS" and APPLE_GPU_AVAILABLE:
            # Use PyTorch MPS for Apple Silicon
            return self._mps_dtw_batch(curvatures_batch)
        else:
            return self._cpu_dtw_batch(curvatures_batch)
    
    def _cuda_dtw_batch(self, curvatures_batch: List[np.ndarray]) -> np.ndarray:
        """CUDA-accelerated DTW computation"""
        n = len(curvatures_batch)
        distance_matrix = cp.zeros((n, n))
        
        # Convert all curvatures to GPU
        gpu_curves = [cp.asarray(curve) for curve in curvatures_batch]
        
        # Parallel DTW computation on GPU
        for i in range(n):
            for j in range(i+1, n):
                # Simplified DTW distance on GPU
                dist = cp.mean(cp.abs(gpu_curves[i] - gpu_curves[j]))
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return distance_matrix.get()  # Return to CPU
    
    def _mps_dtw_batch(self, curvatures_batch: List[np.ndarray]) -> np.ndarray:
        """Apple Silicon MPS-accelerated DTW computation"""
        n = len(curvatures_batch)
        device = torch.device('mps')
        
        # Pad sequences to same length for batch processing
        max_len = max(len(curve) for curve in curvatures_batch)
        padded_curves = []
        
        for curve in curvatures_batch:
            padded = np.pad(curve, (0, max_len - len(curve)), mode='constant')
            padded_curves.append(torch.from_numpy(padded).float().to(device))
        
        # Stack into batch tensor
        batch_tensor = torch.stack(padded_curves)  # Shape: (n, max_len)
        
        # Compute pairwise distances using broadcasting
        diff = batch_tensor.unsqueeze(1) - batch_tensor.unsqueeze(0)  # Shape: (n, n, max_len)
        distances = torch.mean(torch.abs(diff), dim=2)  # Shape: (n, n)
        
        return distances.cpu().numpy()
    
    def _cpu_dtw_batch(self, curvatures_batch: List[np.ndarray]) -> np.ndarray:
        """CPU-based DTW computation (fallback)"""
        n = len(curvatures_batch)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                try:
                    dtw_dist, _ = fastdtw(
                        curvatures_batch[i].reshape(-1, 1), 
                        curvatures_batch[j].reshape(-1, 1), 
                        dist=euclidean
                    )
                    avg_length = (len(curvatures_batch[i]) + len(curvatures_batch[j])) / 2
                    normalized_dist = dtw_dist / avg_length if avg_length > 0 else float('inf')
                    distance_matrix[i, j] = normalized_dist
                    distance_matrix[j, i] = normalized_dist
                except:
                    distance_matrix[i, j] = float('inf')
                    distance_matrix[j, i] = float('inf')
        
        return distance_matrix

    def check_gpu_performance(self) -> Dict[str, Any]:
        """
        Check GPU availability and performance for section comparison
        """
        performance_info = {
            'gpu_available': self.use_gpu,
            'gpu_type': self.gpu_type,
            'performance_comparison': {}
        }
        
        if not self.use_gpu:
            logger.info("💻 GPU acceleration not available - using CPU only")
            return performance_info
        
        # Test performance with sample data
        # logger.info("🧪 Testing GPU vs CPU performance...")
        
        # Create sample curvature data
        sample_size = 50
        sample_curves = [np.random.randn(20) for _ in range(sample_size)]
        
        # Test CPU performance
        start_time = time.time()
        cpu_result = self._cpu_dtw_batch(sample_curves)
        cpu_time = time.time() - start_time
        
        # Test GPU performance
        start_time = time.time()
        gpu_result = self._gpu_dtw_batch(sample_curves)
        gpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
        
        performance_info['performance_comparison'] = {
            'cpu_time': round(cpu_time, 4),
            'gpu_time': round(gpu_time, 4),
            'speedup': round(speedup, 2),
            'sample_size': sample_size
        }
        
        if speedup > 1.5:
            pass
            # logger.info(f"🚀 GPU acceleration effective: {speedup:.2f}x speedup")
        elif speedup > 1.0:
            pass
            # logger.info(f"⚡ GPU acceleration modest: {speedup:.2f}x speedup")
        else:
            pass
            # logger.info(f"⚠️ GPU slower than CPU: {speedup:.2f}x (consider using CPU)")
            
        return performance_info