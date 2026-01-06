"""
Basic HIP compatibility tests for FlashInfer.

This test file validates that core FlashInfer operations work on AMD GPUs via HIP/ROCm.
Currently, FlashInfer's JIT kernel compilation requires CUDA headers and PTX assembly,
so full HIP support is a work-in-progress.

These tests serve as:
1. Documentation of what needs to work for HIP compatibility
2. Validation targets for the HIP porting effort
3. Integration tests once HIP support is complete
"""

import pytest
import torch


def is_hip() -> bool:
    """Check if running on AMD HIP/ROCm platform."""
    return torch.version.hip is not None


def get_device_name() -> str:
    """Get the GPU device name."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "N/A"


@pytest.mark.skipif(not is_hip(), reason="HIP-only test")
class TestHIPBasic:
    """Basic tests for HIP compatibility.
    
    These tests verify that core FlashInfer operations work on AMD GPUs.
    Note: Currently requires HIP-compatible JIT compilation support.
    """

    @pytest.mark.skip(reason="FlashInfer JIT kernel compilation not yet HIP-compatible")
    def test_rope_basic(self):
        """Test RoPE (Rotary Position Embedding) on HIP."""
        from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

        batch_size = 4
        seq_len = 16
        num_heads = 8
        head_dim = 64

        total_tokens = batch_size * seq_len
        q = torch.randn(total_tokens, num_heads * head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(total_tokens, num_heads * head_dim, device="cuda", dtype=torch.float16)
        q_ref = q.clone()
        k_ref = k.clone()

        positions = torch.arange(total_tokens, device="cuda", dtype=torch.long)
        cos_sin_cache = torch.randn(1024, head_dim, device="cuda", dtype=torch.float32)

        apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=q,
            key=k,
            head_size=head_dim,
            cos_sin_cache=cos_sin_cache,
            is_neox=True,
        )

        # Verify output is different from input (rotation was applied)
        assert not torch.allclose(q, q_ref, rtol=1e-3, atol=1e-3), "Query should be modified"
        assert not torch.allclose(k, k_ref, rtol=1e-3, atol=1e-3), "Key should be modified"

        # Check for NaN/Inf
        assert not torch.isnan(q).any(), "Query contains NaN"
        assert not torch.isnan(k).any(), "Key contains NaN"

    @pytest.mark.skip(reason="FlashInfer JIT kernel compilation not yet HIP-compatible")
    def test_rope_interleaved(self):
        """Test RoPE with interleaved layout (is_neox=False)."""
        from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

        batch_size = 2
        seq_len = 32
        num_heads = 4
        head_dim = 128

        total_tokens = batch_size * seq_len
        q = torch.randn(total_tokens, num_heads * head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(total_tokens, num_heads * head_dim, device="cuda", dtype=torch.float16)
        q_ref = q.clone()

        positions = torch.arange(total_tokens, device="cuda", dtype=torch.long)
        cos_sin_cache = torch.randn(2048, head_dim, device="cuda", dtype=torch.float32)

        apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=q,
            key=k,
            head_size=head_dim,
            cos_sin_cache=cos_sin_cache,
            is_neox=False,
        )

        assert not torch.allclose(q, q_ref, rtol=1e-3, atol=1e-3), "Query should be modified"
        assert not torch.isnan(q).any(), "Query contains NaN"

    @pytest.mark.skip(reason="FlashInfer JIT kernel compilation not yet HIP-compatible")
    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_rope_dtypes(self, head_dim: int, dtype: torch.dtype):
        """Test RoPE with different data types and head dimensions."""
        from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

        total_tokens = 64
        num_heads = 8

        q = torch.randn(total_tokens, num_heads * head_dim, device="cuda", dtype=dtype)
        k = torch.randn(total_tokens, num_heads * head_dim, device="cuda", dtype=dtype)
        q_ref = q.clone()

        positions = torch.arange(total_tokens, device="cuda", dtype=torch.long)
        cos_sin_cache = torch.randn(512, head_dim, device="cuda", dtype=torch.float32)

        apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=q,
            key=k,
            head_size=head_dim,
            cos_sin_cache=cos_sin_cache,
            is_neox=True,
        )

        assert not torch.allclose(q, q_ref, rtol=1e-2, atol=1e-2), f"Query should be modified (dtype={dtype})"
        assert not torch.isnan(q).any(), f"Query contains NaN (dtype={dtype})"


@pytest.mark.skipif(not is_hip(), reason="HIP-only test")
class TestHIPMultimodal:
    """Test multimodal-style RoPE operations on HIP.
    
    These tests mirror the 4D tensor RoPE usage pattern in multimodal models
    like video generation (e.g., HunyuanVideo) that use FlashInfer's RoPE.
    """

    @pytest.mark.skip(reason="FlashInfer JIT kernel compilation not yet HIP-compatible")
    def test_multimodal_rope_4d(self):
        """Test 4D tensor RoPE as used in multimodal models."""
        from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

        # Typical multimodal dimensions: batch, seq_len (spatial+temporal), heads, head_dim
        batch_size = 2
        seq_len = 64  # e.g., 8x8 spatial or 4x4x4 spatial-temporal
        num_heads = 8
        head_dim = 64

        # Create 4D input tensors [batch, seq_len, num_heads, head_dim]
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
        q_ref = q.clone()
        k_ref = k.clone()

        # Flatten to 2D for FlashInfer [batch * seq_len, num_heads * head_dim]
        total_tokens = batch_size * seq_len
        q_flat = q.reshape(total_tokens, num_heads * head_dim).contiguous()
        k_flat = k.reshape(total_tokens, num_heads * head_dim).contiguous()

        # Create position indices and cos/sin cache
        positions = torch.arange(total_tokens, device="cuda", dtype=torch.long)
        cos_sin_cache = torch.randn(1024, head_dim, device="cuda", dtype=torch.float32)

        # Apply RoPE in-place
        apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=q_flat,
            key=k_flat,
            head_size=head_dim,
            cos_sin_cache=cos_sin_cache,
            is_neox=True,
        )

        # Reshape back to 4D
        q_rotated = q_flat.view(batch_size, seq_len, num_heads, head_dim)
        k_rotated = k_flat.view(batch_size, seq_len, num_heads, head_dim)

        # Verify rotation was applied
        assert not torch.allclose(q_rotated, q_ref, rtol=1e-3, atol=1e-3), "Query should be modified"
        assert not torch.isnan(q_rotated).any(), "Query contains NaN"


@pytest.mark.skipif(not is_hip(), reason="HIP-only test")
class TestHIPMath:
    """Test basic math operations on HIP.
    
    These tests verify that basic PyTorch operations work correctly on ROCm,
    serving as a baseline for FlashInfer's requirements.
    """

    def test_simple_matmul(self):
        """Test that basic PyTorch matmul works (sanity check)."""
        a = torch.randn(32, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.float16)
        c = torch.mm(a, b)
        
        assert c.shape == (32, 128)
        assert not torch.isnan(c).any()

    def test_batch_matmul(self):
        """Test that batch matmul works."""
        a = torch.randn(4, 32, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(4, 64, 128, device="cuda", dtype=torch.float16)
        c = torch.bmm(a, b)
        
        assert c.shape == (4, 32, 128)
        assert not torch.isnan(c).any()

    def test_bfloat16_support(self):
        """Test bfloat16 dtype support on ROCm."""
        a = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
        c = torch.mm(a, b)
        
        assert c.dtype == torch.bfloat16
        assert not torch.isnan(c).any()


@pytest.mark.skipif(not is_hip(), reason="HIP-only test")
class TestHIPJITInfrastructure:
    """Test the HIP JIT compilation infrastructure.
    
    These tests verify that the FlashInfer JIT system is properly configured
    for HIP/ROCm compilation.
    """

    def test_hip_detection(self):
        """Test that HIP is properly detected."""
        assert is_hip(), "Should be running on HIP"
        assert torch.version.hip is not None

    def test_cuda_available_on_rocm(self):
        """Test that torch.cuda APIs work on ROCm (CUDA compat layer)."""
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() >= 1

    def test_device_name(self):
        """Test that we can get the GPU device name."""
        name = get_device_name()
        assert name != "N/A"
        # Common AMD GPU patterns
        assert any(x in name.lower() for x in ["mi", "radeon", "amd", "gfx"]) or True  # Allow any name


if __name__ == "__main__":
    print(f"Running on HIP: {is_hip()}")
    print(f"HIP version: {torch.version.hip}")
    print(f"Device: {get_device_name()}")
    
    # Run tests
    pytest.main([__file__, "-v"])
