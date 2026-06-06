# Contributing — ternary-llm

> *Thank you for contributing to the SuperInstance fleet.*

## How to Contribute

### 1. Understand the Ternary Ethos

We use {-1, 0, +1} because it is the minimum viable alphabet for expressing agreement,
disagreement, and abstention. **0 is not "nothing"** — it is a deliberate neutral state.

### 2. Set Up Your Environment

```bash
git clone https://github.com/SuperInstance/ternary-llm.git
cd ternary-llm
cargo build
cargo test
```

### 3. Find Something to Work On

- Check the [issues page](https://github.com/SuperInstance/ternary-llm/issues)
- Look for `good first issue` labels
- Read the `ARCHITECTURE.md` and `GETTING_STARTED.md` for context

### 4. Code Standards

- **No unsafe code** unless absolutely necessary and reviewed
- **Ternary-compatible** — prefer {-1, 0, +1} semantics where domain-appropriate
- **Test coverage** — every public function needs at least one test
- **Documentation** — all public items must have doc comments
- **Error handling** — use `Result` for fallible operations; `FluxError` pattern preferred

### 5. Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) spec:

```
feat: add ternary quantization for weight matrices
fix: handle zero-input edge case in softmax
docs: update architecture overview for v0.3
test: add roundtrip tests for pack/unpack
```

### 6. Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Write tests for your changes
4. Ensure all tests pass: `cargo test`
5. Run clippy: `cargo clippy -- -D warnings`
6. Submit a PR with a clear description of the change

### 7. Review Process

- All PRs need at least one approval from a maintainer
- Changes to public API require careful review
- Breaking changes must be discussed in an issue first

## Code of Conduct

Be respectful. We're building something complex together. Assume good intent.

## License

By contributing, you agree that your contributions will be licensed under MIT OR Apache-2.0.
