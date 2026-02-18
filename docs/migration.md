# Migration Guide

This guide helps you migrate from other LLM libraries to ChukLLM.

## Migrating from OpenAI SDK

### Before (OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(api_key="sk-...")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```

### After (ChukLLM)
```python
from chuk_llm import ask_openai_sync

# No client initialization needed!
response = ask_openai_sync("Hello")
print(response)
```

## Migrating from LangChain

### Before (LangChain)
```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("colorful socks")
```

### After (ChukLLM)
```python
from chuk_llm import ask_sync

# Direct and simple
result = ask_sync(
    "What is a good name for a company that makes colorful socks?",
    temperature=0.7
)
```

### Conversation Memory

**Before (LangChain)**
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory
)
conversation.run("Hi, my name is Alice")
conversation.run("What's my name?")
```

**After (ChukLLM)**
```python
from chuk_llm import conversation_sync

with conversation_sync() as chat:
    chat.ask("Hi, my name is Alice")
    response = chat.ask("What's my name?")  # Remembers context
```

## Migrating from LiteLLM

### Before (LiteLLM)
```python
from litellm import completion

response = completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}],
    api_key="sk-..."
)
```

### After (ChukLLM)
```python
from chuk_llm import ask_sync

# Auto-detects provider from environment
response = ask_sync("Hello", model="gpt-3.5-turbo")
```

## Migrating from Anthropic SDK

### Before (Anthropic SDK)
```python
from anthropic import Anthropic

client = Anthropic(api_key="sk-ant-...")
message = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello"}]
)
print(message.content[0].text)
```

### After (ChukLLM)
```python
from chuk_llm import ask_claude_sync

# Simple and direct
response = ask_claude_sync("Hello", max_tokens=100)
print(response)
```

## Feature Comparison

| Feature | OpenAI SDK | LangChain | LiteLLM | ChukLLM |
|---------|------------|-----------|---------|---------|
| Simple API | ✅ | ❌ | ✅ | ✅ |
| Multi-provider | ❌ | ✅ | ✅ | ✅ |
| Auto-discovery | ❌ | ❌ | ❌ | ✅ |
| Built-in sessions | ❌ | ❌ | ❌ | ✅ |
| Provider functions | ❌ | ❌ | ❌ | ✅ |
| Streaming | ✅ | ⚠️ | ✅ | ✅ |
| Async support | ✅ | ⚠️ | ✅ | ✅ |
| CLI included | ❌ | ❌ | ⚠️ | ✅ |

## Environment Variables

ChukLLM uses the same environment variable names as the original SDKs:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://..."

# Google
export GOOGLE_API_KEY="..."

# No changes needed to your .env files!
```

## Advanced Migration Topics

### Tool/Function Calling

**OpenAI SDK:**
```python
tools = [{"type": "function", "function": {...}}]
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools
)
```

**ChukLLM:**
```python
# Same tool format!
response = ask_sync("Question", tools=tools)
```

### Streaming

**OpenAI SDK:**
```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

**ChukLLM:**
```python
async for chunk in stream("Question"):
    print(chunk, end="")
```

### Error Handling

ChukLLM provides consistent error handling across all providers:

```python
from chuk_llm import ask_sync

try:
    response = ask_sync("Hello")
except Exception as e:
    # Unified error handling for all providers
    print(f"Error: {e}")
```

## Getting Help

If you need help migrating:

1. Check the [examples folder](../examples/)
2. Ask in [GitHub Discussions](https://github.com/chrishayuk/chuk-llm/discussions)
3. Open an [issue](https://github.com/chrishayuk/chuk-llm/issues)

## Why Migrate?

- **Simpler API**: Less boilerplate, more productivity
- **Auto-discovery**: New models work instantly
- **Built-in analytics**: Track costs and usage automatically
- **Unified interface**: Same API for all providers
- **Better DX**: Great CLI, sensible defaults, clear errors