#!/usr/bin/env python3
"""Test vLLM server connection and structured output."""

import asyncio
import json
from openai import AsyncOpenAI

async def test_basic_completion():
    """Test basic chat completion."""
    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        timeout=60.0,
    )

    print("=" * 60)
    print("Test 1: Basic Chat Completion")
    print("=" * 60)

    try:
        response = await client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "user", "content": "Hello! What is 2+2? Answer briefly."}
            ],
            max_tokens=50,
            temperature=0.7,
        )

        print(f"✅ Success!")
        print(f"Response: {response.choices[0].message.content}")
        print(f"Tokens: {response.usage.completion_tokens}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


async def test_json_mode():
    """Test JSON mode (structured output)."""
    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        timeout=60.0,
    )

    print("\n" + "=" * 60)
    print("Test 2: JSON Mode (Structured Output)")
    print("=" * 60)

    try:
        response = await client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": """You must respond with valid JSON.

Respond to this question in JSON format with these fields:
- answer: your answer
- confidence: a number from 0-100

Question: What is the capital of France?"""
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=100,
            temperature=0.7,
        )

        content = response.choices[0].message.content
        print(f"✅ Success!")
        print(f"Response: {content}")

        # Try to parse JSON
        try:
            data = json.loads(content)
            print(f"✅ Valid JSON!")
            print(f"Parsed data: {json.dumps(data, indent=2)}")
            return True
        except json.JSONDecodeError as e:
            print(f"⚠️  Response is not valid JSON: {e}")
            return False

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n🚀 Testing vLLM Server Connection")
    print(f"Server: http://localhost:8000")
    print(f"Model: openai/gpt-oss-20b\n")

    # Test 1: Basic completion
    test1_passed = await test_basic_completion()

    # Test 2: JSON mode
    test2_passed = await test_json_mode()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test 1 (Basic Completion): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (JSON Mode): {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Server is ready for ARC experiments!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the logs above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
