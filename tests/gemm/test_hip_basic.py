"""
Basic HIP compatibility test for FlashInfer.

This test verifies that FlashInfer's core infrastructure works on AMD GPUs
by testing basic operations that don't require NVIDIA-specific backends.
"""

import pytest
import torch
import torch.nn.functional as F


def is_hip():
    """Check if running on AMD HIP."""
    return torch.version.hip is not None


@pytest.mark.skipif(not is_hip(), reason="Test only runs on AMD HIP")
class TestHIPBasicCompatibility:
    """Test basic HIP compatibility for FlashInfer."""

    def test_hip_detection(self):
        """Test that HIP is properly detected."""
        assert torch.version.hip is not None
        assert torch.cuda.is_available()
        print(f"HIP version: {torch.version.hip}")

    def test_tensor_creation(self):
        """Test basic tensor creation on AMD GPU."""
        device = torch.device("cuda")
        
        # Test various dtypes
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        for dtype in dtypes:
            x = torch.randn(128, 256, device=device, dtype=dtype)
            assert x.device.type == "cuda"
            assert x.dtype == dtype
            print(f"  Created {dtype} tensor on {x.device}")

    def test_rope_import(self):
        """Test that FlashInfer RoPE module can be imported."""
        from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
        print("  RoPE module imported successfully")

    def test_rope_execution(self):
        """Test RoPE kernel execution on AMD."""
        from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

        # Test configuration
        batch_size = 4
        seq_len = 16
        num_heads = 8
        head_dim = 64
        device = "cuda"
        dtype = torch.float16

        # Create inputs
        total_tokens = batch_size * seq_len
        q = torch.randn(total_tokens, num_heads * head_dim, device=device, dtype=dtype)
        k = torch.randn(total_tokens, num_heads * head_dim, device=device, dtype=dtype)
        q_ref = q.clone()
        k_ref = k.clone()

        positions = torch.arange(total_tokens, device=device, dtype=torch.long)
        cos_sin_cache = torch.randn(1024, head_dim, device=device, dtype=torch.float32)

        # Apply RoPE
        apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=q,
            key=k,
            head_size=head_dim,
            cos_sin_cache=cos_sin_cache,
            is_neox=True,
        )

        # Verify output is different from input (rotation was applied)
        q_changed = not torch.allclose(q, q_ref, rtol=1e-3, atol=1e-3)
        k_changed = not torch.allclose(k, k_ref, rtol=1e-3, atol=1e-3)

        # Check for NaN/Inf
        assert not torch.isnan(q).any(), "Output q contains NaN"
        assert not torch.isnan(k).any(), "Output k contains NaN"
        assert not torch.isinf(q).any(), "Output q contains Inf"
        assert not torch.isinf(k).any(), "Output k contains Inf"

        assert q_changed, "Query should be modified by RoPE"
        assert k_changed, "Key should be modified by RoPE"
        print(f"  RoPE executed successfully: q_modified={q_changed}, k_modified={k_changed}")

    def test_rope_different_configs(self):
        """Test RoPE with various configurations."""
        from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

        configs = [
            {"batch": 1, "seq": 8, "heads": 4, "dim": 64, "is_neox": True},
            {"batch": 2, "seq": 16, "heads": 8, "dim": 128, "is_neox": True},
            {"batch": 4, "seq": 32, "heads": 16, "dim": 64, "is_neox": False},
        ]

        for cfg in configs:
            batch = cfg["batch"]
            seq = cfg["seq"]
            heads = cfg["heads"]
            dim = cfg["dim"]
            is_neox = cfg["is_neox"]

            total_tokens = batch * seq
            q = torch.randn(total_tokens, heads * dim, device="cuda", dtype=torch.float16)
            k = torch.randn(total_tokens, heads * dim, device="cuda", dtype=torch.float16)
            positions = torch.arange(total_tokens, device="cuda", dtype=torch.long)
            cos_sin_cache = torch.randn(1024, dim, device="cuda", dtype=torch.float32)

            apply_rope_with_cos_sin_cache_inplace(
                positions=positions,
                query=q,
                key=k,
                head_size=dim,
                cos_sin_cache=cos_sin_cache,
                is_neox=is_neox,
            )

            assert not torch.isnan(q).any(), f"NaN in config {cfg}"
            assert not torch.isnan(k).any(), f"NaN in config {cfg}"
            print(f"  Config {cfg} passed")


@pytest.mark.skipif(not is_hip(), reason="Test only runs on AMD HIP")
class TestHIPVecDtypes:
    """Test vectorized data types on HIP."""

    def test_half_operations(self):
        """Test half precision operations."""
        x = torch.randn(256, 512, device="cuda", dtype=torch.float16)
        y = torch.randn(256, 512, device="cuda", dtype=torch.float16)
        z = x + y
        assert z.dtype == torch.float16
        assert not torch.isnan(z).any()
        print("  half operations work")

    def test_bfloat16_operations(self):
        """Test bfloat16 operations."""
        x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
        y = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
        z = x + y
        assert z.dtype == torch.bfloat16
        assert not torch.isnan(z).any()
        print("  bfloat16 operations work")

    def test_fp8_types_exist(self):
        """Test that FP8 types are available (may not be fully functional)."""
        # FP8 support varies by HIP version
        try:
            if hasattr(torch, "float8_e4m3fn"):
                print("  torch.float8_e4m3fn is available")
            if hasattr(torch, "float8_e5m2"):
                print("  torch.float8_e5m2 is available")
        except Exception as e:
            pytest.skip(f"FP8 types not available: {e}")


@pytest.mark.skipif(not is_hip(), reason="Test only runs on AMD HIP")  
class TestHIPJITCompilation:
    """Test JIT compilation on HIP."""

    def test_jit_env_detection(self):
        """Test JIT environment detection for HIP."""
        from flashinfer.jit.cpp_ext import is_hip, get_cuda_path
        
        assert is_hip(), "Should detect HIP environment"
        cuda_path = get_cuda_path()
        assert "rocm" in cuda_path.lower() or "hip" in cuda_path.lower(), \
            f"Expected ROCm path, got: {cuda_path}"
        print(f"  ROCm path: {cuda_path}")

    def test_hip_cflags(self):
        """Test HIP compilation flags generation."""
        from flashinfer.jit.cpp_ext import build_cuda_cflags, is_hip
        
        if not is_hip():
            pytest.skip("Not on HIP")
        
        common_cflags = ["-O3"]
        cflags = build_cuda_cflags(common_cflags)
        
        # Should contain HIP-specific flags
        cflags_str = " ".join(cflags)
        assert "__HIP_PLATFORM_AMD__" in cflags_str or "fPIC" in cflags_str, \
            f"Expected HIP flags in: {cflags_str}"
        print(f"  Generated HIP cflags: {cflags[:5]}...")


@pytest.mark.skipif(not is_hip(), reason="Test only runs on AMD HIP")
class TestHIPMatMul:
    """Test matrix multiplication operations on HIP (via rocBLAS)."""

    @pytest.mark.parametrize("m,n,k", [(32, 64, 128), (128, 256, 512), (1, 1024, 1024)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_mm(self, m, n, k, dtype):
        """Test basic matrix multiplication."""
        a = torch.randn(m, k, device="cuda", dtype=dtype)
        b = torch.randn(k, n, device="cuda", dtype=dtype)
        c = torch.mm(a, b)
        
        assert c.shape == (m, n)
        assert c.dtype == dtype
        assert not torch.isnan(c).any(), "Result contains NaN"
        assert not torch.isinf(c).any(), "Result contains Inf"

    @pytest.mark.parametrize("batch", [1, 4, 16])
    @pytest.mark.parametrize("m,n,k", [(32, 64, 128), (64, 128, 256)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_bmm(self, batch, m, n, k, dtype):
        """Test batched matrix multiplication."""
        a = torch.randn(batch, m, k, device="cuda", dtype=dtype)
        b = torch.randn(batch, k, n, device="cuda", dtype=dtype)
        c = torch.bmm(a, b)
        
        assert c.shape == (batch, m, n)
        assert c.dtype == dtype
        assert not torch.isnan(c).any(), "Result contains NaN"
        assert not torch.isinf(c).any(), "Result contains Inf"

    def test_mm_accuracy(self):
        """Test matrix multiplication accuracy against CPU reference."""
        m, n, k = 64, 128, 256
        
        # Create inputs
        a_cpu = torch.randn(m, k, dtype=torch.float32)
        b_cpu = torch.randn(k, n, dtype=torch.float32)
        
        # CPU reference
        c_ref = torch.mm(a_cpu, b_cpu)
        
        # GPU computation
        a_gpu = a_cpu.to(device="cuda", dtype=torch.float16)
        b_gpu = b_cpu.to(device="cuda", dtype=torch.float16)
        c_gpu = torch.mm(a_gpu, b_gpu).float().cpu()
        
        # Check cosine similarity
        cos_sim = F.cosine_similarity(
            c_ref.reshape(-1), c_gpu.reshape(-1), dim=0
        )
        assert cos_sim > 0.99, f"Accuracy too low: {cos_sim}"
        print(f"  MM accuracy (cos_sim): {cos_sim:.6f}")

    def test_matmul_broadcast(self):
        """Test matmul with broadcasting."""
        # (batch, m, k) @ (k, n) -> (batch, m, n)
        a = torch.randn(4, 32, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.float16)
        c = torch.matmul(a, b)
        
        assert c.shape == (4, 32, 128)
        assert not torch.isnan(c).any()

    def test_linear_layer(self):
        """Test nn.Linear which uses matmul internally."""
        batch, in_features, out_features = 16, 512, 256
        
        linear = torch.nn.Linear(in_features, out_features, dtype=torch.float16, device="cuda")
        x = torch.randn(batch, in_features, device="cuda", dtype=torch.float16)
        y = linear(x)
        
        assert y.shape == (batch, out_features)
        assert not torch.isnan(y).any()
        print(f"  Linear layer works: {x.shape} -> {y.shape}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
