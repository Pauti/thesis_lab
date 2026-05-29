#!/usr/bin/env python3
"""
Empirical test: What batch sizes can UNETR handle on RTX 4080 16GB?

Tests different batch sizes and measures:
1. Peak VRAM usage
2. Whether training succeeds or OOMs
3. Training speed per batch

This provides scientific evidence for our batch size choice.

Usage:
    python scripts/test_batch_size_memory.py
"""

import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.unetr import get_unetr_model
from utils.losses import get_loss


def test_batch_size(batch_size: int, patch_size: tuple, device: str = "cuda"):
    """Test if a batch size fits in memory and measure peak VRAM."""
    print(f"\n{'='*70}")
    print(f"Testing batch_size={batch_size}")
    print(f"{'='*70}")
    
    # Create model
    model = get_unetr_model(
        img_size=patch_size,
        in_channels=4,
        out_channels=4,
        dropout_rate=0.0,
    ).to(device)
    
    criterion = get_loss(include_background=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Reset CUDA memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    initial_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"Initial VRAM (model only): {initial_mem:.2f} GB")
    
    try:
        # Create synthetic batch
        imgs = torch.randn(batch_size, 4, *patch_size, device=device)
        lbls = torch.randint(0, 4, (batch_size, 1, *patch_size), device=device)
        
        print(f"Input batch shape: {imgs.shape}")
        print(f"Allocated after input: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Forward pass
        model.train()
        t0 = time.time()
        logits = model(imgs)
        forward_time = time.time() - t0
        
        forward_mem = torch.cuda.memory_allocated() / 1024**3
        forward_peak = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"After forward pass:")
        print(f"  Allocated: {forward_mem:.2f} GB")
        print(f"  Peak:      {forward_peak:.2f} GB")
        print(f"  Time:      {forward_time:.3f} sec")
        
        # Backward pass
        loss = criterion(logits, lbls)
        
        t0 = time.time()
        loss.backward()
        backward_time = time.time() - t0
        
        backward_mem = torch.cuda.memory_allocated() / 1024**3
        backward_peak = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"After backward pass:")
        print(f"  Allocated: {backward_mem:.2f} GB")
        print(f"  Peak:      {backward_peak:.2f} GB")
        print(f"  Time:      {backward_time:.3f} sec")
        
        # Optimizer step
        t0 = time.time()
        optimizer.step()
        optimizer.zero_grad()
        optimizer_time = time.time() - t0
        
        optimizer_mem = torch.cuda.memory_allocated() / 1024**3
        optimizer_peak = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"After optimizer step:")
        print(f"  Allocated: {optimizer_mem:.2f} GB")
        print(f"  Peak:      {optimizer_peak:.2f} GB")
        print(f"  Time:      {optimizer_time:.3f} sec")
        
        total_time = forward_time + backward_time + optimizer_time
        
        print(f"\n✅ SUCCESS!")
        print(f"Total time: {total_time:.3f} sec")
        print(f"Peak VRAM:  {optimizer_peak:.2f} GB")
        
        # Cleanup
        del imgs, lbls, logits, loss
        torch.cuda.empty_cache()
        
        return {
            "batch_size": batch_size,
            "success": True,
            "peak_vram_gb": optimizer_peak,
            "forward_time": forward_time,
            "backward_time": backward_time,
            "optimizer_time": optimizer_time,
            "total_time": total_time,
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n💥 OUT OF MEMORY!")
            print(f"Error: {str(e)[:200]}")
            
            # Try to get peak before OOM
            try:
                peak_mem = torch.cuda.max_memory_allocated() / 1024**3
                print(f"Peak VRAM before OOM: {peak_mem:.2f} GB")
            except:
                peak_mem = None
            
            torch.cuda.empty_cache()
            
            return {
                "batch_size": batch_size,
                "success": False,
                "peak_vram_gb": peak_mem,
                "error": str(e)[:200],
            }
        else:
            raise


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("ERROR: CUDA not available!")
        return
    
    print("="*70)
    print("UNETR Batch Size Memory Test")
    print("="*70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    patch_size = (128, 128, 128)
    batch_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
    
    print(f"Testing batch sizes: {batch_sizes}")
    print(f"Patch size: {patch_size}")
    print()
    
    results = []
    
    for batch_size in batch_sizes:
        result = test_batch_size(batch_size, patch_size, device)
        results.append(result)
        
        # Stop if we hit OOM
        if not result["success"]:
            print(f"\n⚠️  Stopped testing at batch_size={batch_size} (OOM)")
            break
        
        # Small delay
        time.sleep(2)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print()
    
    print(f"{'Batch':<8} {'Success':<10} {'Peak VRAM':<12} {'Total Time':<12}")
    print("-" * 70)
    
    for r in results:
        success = "✅ YES" if r["success"] else "❌ OOM"
        peak = f"{r['peak_vram_gb']:.2f} GB" if r["peak_vram_gb"] else "N/A"
        total = f"{r.get('total_time', 0):.3f} sec" if r["success"] else "N/A"
        print(f"{r['batch_size']:<8} {success:<10} {peak:<12} {total:<12}")
    
    print()
    print("="*70)
    print("ANALYSIS")
    print("="*70)
    print()
    
    # Find max successful batch size
    successful = [r for r in results if r["success"]]
    
    if successful:
        max_batch = max(r["batch_size"] for r in successful)
        max_batch_result = [r for r in successful if r["batch_size"] == max_batch][0]
        
        print(f"✅ Maximum batch size that fits: {max_batch}")
        print(f"   Peak VRAM: {max_batch_result['peak_vram_gb']:.2f} GB")
        print(f"   Time per iteration: {max_batch_result['total_time']:.3f} sec")
        print()
        
        # Compare to our choice (batch=2, grad_acc=4)
        batch2_result = [r for r in successful if r["batch_size"] == 2][0]
        
        print(f"Our choice (batch=2):")
        print(f"   Peak VRAM: {batch2_result['peak_vram_gb']:.2f} GB")
        print(f"   Time per iteration: {batch2_result['total_time']:.3f} sec")
        print()
        
        # Calculate effective batch=8 with grad_acc
        effective_batch_time = batch2_result['total_time'] * 4  # 4 accumulations
        
        print(f"With gradient accumulation (batch=2, grad_acc=4):")
        print(f"   Peak VRAM: {batch2_result['peak_vram_gb']:.2f} GB (same)")
        print(f"   Time per effective batch: {effective_batch_time:.3f} sec")
        print(f"   Effective batch size: 8")
        print()
        
        # Compare to direct batch=8
        if max_batch >= 8:
            batch8_result = [r for r in successful if r["batch_size"] == 8][0]
            print(f"Direct batch=8 (if it fits):")
            print(f"   Peak VRAM: {batch8_result['peak_vram_gb']:.2f} GB")
            print(f"   Time per iteration: {batch8_result['total_time']:.3f} sec")
            print(f"   Speedup vs. grad_acc: {effective_batch_time / batch8_result['total_time']:.2f}x")
        else:
            print(f"❌ Direct batch=8 does NOT fit! (OOM)")
            print(f"   → Gradient accumulation is NECESSARY")
        
        print()
        
        # Safety margin analysis
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        vram_used = batch2_result['peak_vram_gb']
        vram_free = vram_total - vram_used
        safety_margin = vram_free / vram_total * 100
        
        print(f"Safety margin with batch=2:")
        print(f"   Total VRAM: {vram_total:.2f} GB")
        print(f"   Used: {vram_used:.2f} GB ({vram_used/vram_total*100:.1f}%)")
        print(f"   Free: {vram_free:.2f} GB ({safety_margin:.1f}%)")
        print()
        
        if safety_margin > 30:
            print(f"✅ Good safety margin ({safety_margin:.1f}%)")
        elif safety_margin > 15:
            print(f"⚠️  Moderate safety margin ({safety_margin:.1f}%)")
        else:
            print(f"❌ Low safety margin ({safety_margin:.1f}%) - risk of OOM spikes!")
    
    print()
    print("="*70)
    print("CONCLUSION")
    print("="*70)
    print()
    
    # Save results
    import json
    output_file = Path("experiments/batch_size_memory_test.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
