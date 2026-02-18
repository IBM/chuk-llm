# Contributing to ChukLLM

Thank you for your interest in contributing to ChukLLM! We welcome contributions of all kinds, from bug fixes to new features.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/chrishayuk/chuk-llm.git
   cd chuk-llm
   ```
3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   # or with uv
   uv sync --dev
   ```

## Development Setup

### Environment Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/api/test_conversation.py

# Run with coverage
make test-cov
```

### Code Quality

```bash
# Run linting
make lint

# Format code
make format

# Type checking
make typecheck

# Run all checks
make check
```

## Contribution Guidelines

### Code Style

- We use `ruff` for linting and formatting
- Follow PEP 8 with a line length of 100 characters
- Use type hints for all function signatures
- Write docstrings for all public functions and classes

### Testing

- Write tests for all new features
- Maintain or improve code coverage
- Test files should mirror the source structure in `tests/`
- Use `pytest` fixtures for common test setup

### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or fixes
- `refactor`: Code refactoring
- `style`: Code style changes
- `perf`: Performance improvements
- `chore`: Maintenance tasks

Examples:
```
feat(conversation): add branching support for conversations
fix(ollama): handle connection errors gracefully
docs(readme): simplify quick start section
```

### Pull Request Process

1. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit them with clear messages

3. **Add tests** for new functionality

4. **Run checks** locally:
   ```bash
   make check
   ```

5. **Push to your fork** and create a pull request

6. **Describe your changes** in the PR:
   - What problem does it solve?
   - What is the approach?
   - Any breaking changes?

### Adding New Providers

#### For OpenAI-Compatible Providers

Most modern LLM APIs follow the OpenAI format. Before implementing a custom client, determine the provider's capabilities:

**Step 1: Test the API**

Use the debug script to understand what the API supports:

```bash
# Test function calling capabilities
python examples/debug/debug_openai_compatible_function_calling.py \
    --provider yourprovider \
    --model your-model-name
```

The script will test:
- Native OpenAI tools support
- Legacy functions parameter
- JSON mode for function calling
- Tool result message formats (tool/user/function roles)

**Step 2: Choose Implementation**

Based on debug results:

**Option A: Native OpenAI Support (simplest)**
```python
# src/chuk_llm/llm/providers/yourprovider_client.py
from .openai_client import OpenAILLMClient

class YourProviderClient(OpenAILLMClient):
    """Simple wrapper - API fully supports OpenAI format"""
    pass
```

**Option B: JSON Function Calling Fallback**

If the API accepts tools but doesn't call them natively (debug script will tell you):

```python
from .openai_client import OpenAILLMClient

class YourProviderClient(OpenAILLMClient):
    """Uses JSON fallback for function calling"""

    # Enable JSON function calling fallback
    ENABLE_JSON_FUNCTION_FALLBACK = True
    SUPPORTS_TOOL_ROLE = False  # Set based on debug results
    SUPPORTS_FUNCTION_ROLE = False  # Set based on debug results

    def __init__(self, model: str, api_key: str, api_base: str | None = None, **kwargs):
        super().__init__(model, api_key, api_base, **kwargs)
        self.detected_provider = "yourprovider"
```

**Step 3: Add Configuration**

Add to `src/chuk_llm/chuk_llm.yaml`:

```yaml
yourprovider:
  client_class: "chuk_llm.llm.providers.yourprovider_client:YourProviderClient"
  api_key_env: "YOURPROVIDER_API_KEY"
  api_base: "https://api.yourprovider.com/v1"
  default_model: "your-default-model"
  models: ["*"]
```

**Step 4: Add Examples**

Create `examples/providers/yourprovider_usage_examples.py` following existing patterns.

**Step 5: Test**

```bash
# Run your examples
python examples/providers/yourprovider_usage_examples.py

# Run tests
pytest tests/llm/providers/test_yourprovider_client.py
```

See `examples/debug/README.md` for detailed debug script documentation.

#### For Custom Providers

For providers with completely different APIs (non-OpenAI-compatible):

1. Create a new client in `src/chuk_llm/llm/providers/`
2. Inherit from `BaseLLMClient`
3. Implement required methods: `create_completion()`, streaming support
4. Add provider configuration to `chuk_llm.yaml`
5. Add tests in `tests/llm/providers/`
6. Update documentation

Example structure:
```python
# src/chuk_llm/llm/providers/newprovider_client.py
from chuk_llm.llm.core.base import BaseLLMClient

class NewProviderClient(BaseLLMClient):
    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(provider="newprovider", **kwargs)
        # Initialize client

    async def create_completion(self, messages, tools=None, **kwargs):
        # Implement completion logic
        pass
```

### Adding New Features

Before adding a major feature:

1. **Open an issue** to discuss the feature
2. **Get feedback** from maintainers
3. **Design the API** with examples
4. **Implement incrementally** with tests

## Project Structure

```
chuk-llm/
├── src/chuk_llm/
│   ├── api/           # Public API layer
│   ├── llm/           # LLM client implementations
│   ├── configuration/ # Config management
│   └── cli.py         # CLI implementation
├── tests/             # Test suite
├── examples/          # Usage examples
├── docs/              # Documentation
└── benchmarks/        # Performance benchmarks
```

## Getting Help

- 💬 [Discussions](https://github.com/chrishayuk/chuk-llm/discussions) - Ask questions
- 🐛 [Issues](https://github.com/chrishayuk/chuk-llm/issues) - Report bugs

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- GitHub contributors page

Thank you for helping make ChukLLM better! 🎉