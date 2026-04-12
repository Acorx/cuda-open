# Contributing to CUDA Open

Thank you for your interest in contributing to CUDA Open! 🚀

## 🚀 Quick Start

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/cuda-open.git
cd cuda-open

# Install in dev mode
pip install -e .

# Run quick test (lightweight!)
python3 -c "
import numpy as np
from cuda_open.quantizer import BitNetQuantizer
data = np.random.randn(100).astype(np.float32)
packed, _ = BitNetQuantizer.quantize(data)
print(f'✓ Compression: {data.nbytes/packed.nbytes:.1f}x')
"
```

## ⚠️ IMPORTANT: Resource Limits

To avoid crashing your PC:

| Limit | Value |
|-------|-------|
| Population MAX | 20 |
| Generations MAX | 10 |
| Timeout | 30 seconds |
| Model MAX | gpt2 (124M) |
| Batch size MAX | 2 |

**Always ask before running anything heavy!**

## 🧪 Testing

```bash
# Test quantizer (lightweight)
python3 -c "
import numpy as np
from cuda_open.quantizer import BitNetQuantizer

data = np.random.randn(100).astype(np.float32)
packed, scale = BitNetQuantizer.quantize(data)
assert packed.nbytes < data.nbytes
print('✓ Test passed')
"

# C++ tests
mkdir build && cd build
cmake .. && make -j$(nproc)
ctest --output-on-failure
```

## 📝 Pull Request Process

1. **Fork** the repo
2. **Create branch**: `git checkout -b feature/my-feature`
3. **Make changes** (keep it lightweight!)
4. **Test** your changes
5. **Commit**: `git commit -m "feat: add my feature"`
6. **Push**: `git push origin feature/my-feature`
7. **Open PR** on GitHub

## 💡 What We Need Help With

### Priority 1: Code Quality
- [ ] Type hints everywhere
- [ ] Docstrings for all functions
- [ ] Unit tests for PyTorch module

### Priority 2: Benchmarks
- [ ] Benchmark vs PyTorch FP32
- [ ] Benchmark vs vLLM
- [ ] Benchmark vs TGI

### Priority 3: Documentation
- [ ] English translations
- [ ] Video tutorials
- [ ] Colab notebooks

### Priority 4: Features
- [ ] CUDA kernel integration
- [ ] Multi-GPU support
- [ ] Model zoo (pre-quantized models)

## 🎯 Good First Issues

Look for issues labeled `good-first-issue` on GitHub!

## 💬 Questions?

- **GitHub Issues**: For bugs and features
- **Discussions**: For questions
- **Email**: team@cuda-open.dev (future)

---

Thanks for contributing! 🎉
