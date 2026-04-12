# Contributing to CUDA Open 🤝

Thank you for your interest! This guide will help you get started.

---

## ⚡ Quick Start

```bash
# Fork and clone
git clone https://github.com/Acorx/cuda-open.git
cd cuda-open

# Install (lightweight!)
pip install numpy
```

---

## ⚠️ IMPORTANT: Resource Limits

To ensure all developers can contribute without crashing their machines:

| Limit | Value | Why |
|-------|-------|-----|
| Population MAX | 20 | Avoid long evolutions |
| Generations MAX | 10 | Keep runs under 30s |
| Timeout | 30 seconds | Prevent hangs |
| Model MAX | gpt2 (124M) | Fit in 2GB RAM |
| Batch size MAX | 2 | Memory safety |

**Always test with small configurations first!**

---

## 🧪 Running Tests

### Python Tests (No PyTorch required!)

```bash
# Test quantizer
python3 -c "
from cuda_open.quantizer import BitNetQuantizer
import numpy as np

data = np.random.randn(100).astype(np.float32)
packed, scale = BitNetQuantizer.quantize(data)
assert packed.nbytes < data.nbytes, 'Compression failed'
print(f'✓ Compression: {data.nbytes/packed.nbytes:.1f}x')
"

# Run full demo
python3 demo.py

# Run benchmark
python3 benchmark/benchmark_light.py
```

### C++ Tests

```bash
mkdir build && cd build
cmake .. -DCUDA_OPEN_BUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```

---

## 📝 Code Style

### Python
- Use `numpy` for numerical operations
- Type hints encouraged: `def foo(x: np.ndarray) -> np.ndarray:`
- Keep imports minimal (lazy imports for heavy deps)

### C++
- C++17 standard
- Use `snake_case` for functions, `PascalCase` for classes
- Add comments for complex algorithms

---

## 🎯 Good First Issues

Look for issues labeled `good-first-issue` on GitHub! Examples:

- [ ] Add docstrings to existing functions
- [ ] Write unit tests for `quantizer.py`
- [ ] Improve README translations
- [ ] Add type hints

---

## 🚀 Pull Request Process

1. **Create a branch**: `git checkout -b feature/my-feature`
2. **Make changes** (keep it lightweight!)
3. **Test your changes** (use commands above)
4. **Commit**: `git commit -m "feat: add my feature"`
5. **Push**: `git push origin feature/my-feature`
6. **Open a PR** on GitHub

---

## 💡 What We Need Help With

### High Priority
- [ ] CUDA kernel optimization
- [ ] Real model benchmarks (7B)
- [ ] Type hints everywhere
- [ ] Unit tests

### Medium Priority
- [ ] Documentation translations (FR→EN)
- [ ] Video tutorials
- [ ] Colab notebooks

### Low Priority
- [ ] Website/landing page
- [ ] Logo design
- [ ] Community management

---

## 💬 Questions?

- **GitHub Issues**: For bugs and features
- **GitHub Discussions**: For questions
- **Email**: arthur@cuda-open.dev

---

Thanks for contributing! 🎉
