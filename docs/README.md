# ChukLLM Documentation

Welcome to the ChukLLM documentation! 

## Documentation Structure

- **[Quick Start](../README.md)** - Get up and running quickly
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Examples](../examples/)** - Code examples and tutorials
- **[Migration Guide](migration.md)** - Migrating from other libraries
- **[Benchmarks](benchmarks.md)** - Performance comparisons
- **[Architecture](architecture.md)** - System design and internals

## Key Concepts

### Providers
ChukLLM supports multiple LLM providers through a unified interface. Each provider has its own client implementation that handles the specific API requirements.

### Auto-Discovery
The discovery system automatically detects available models (especially for Ollama) and generates convenience functions at runtime.

### Session Tracking
Built-in session management tracks all LLM interactions for analytics, cost monitoring, and debugging.

### Conversation Management
Stateful conversations with memory, branching, and persistence support.

## Getting Help

- 💬 [GitHub Discussions](https://github.com/chrishayuk/chuk-llm/discussions)
- 🐛 [Issue Tracker](https://github.com/chrishayuk/chuk-llm/issues)

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing to ChukLLM.