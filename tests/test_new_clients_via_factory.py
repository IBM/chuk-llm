#!/usr/bin/env python3
"""
Test that get_client() factory returns new modern clients
"""
import os
import sys

# Add src to path

from chuk_llm.llm.client import get_client

print("Testing that factory returns new modern clients...\n")

# Test providers (without actually calling APIs - just verify client type)
test_providers = [
    ("openai", "gpt-4"),
    ("groq", "llama-3.3-70b-versatile"),
    ("mistral", "mistral-large-latest"),
]

for provider, model in test_providers:
    try:
        # Set a dummy API key to avoid errors (won't actually call API)
        os.environ[f"{provider.upper()}_API_KEY"] = "test-key-not-real"

        # Get client from factory
        client = get_client(provider, model=model)

        # Check client type
        client_class = client.__class__.__name__
        client_module = client.__class__.__module__

        is_new_client = "chuk_llm.clients" in client_module
        status = "✅ NEW" if is_new_client else "❌ OLD"

        print(f"{status} {provider:12} -> {client_class:30} ({client_module})")

    except Exception as e:
        print(f"❌ {provider:12} -> ERROR: {e}")

print("\n" + "="*80)
print("Summary:")
print("  If all show '✅ NEW' then factory is using new modern clients")
print("  If any show '❌ OLD' then config update didn't work")
print("="*80)
