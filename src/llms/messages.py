import os
from copy import deepcopy

from anthropic import AsyncAnthropic
from devtools import debug
from google import genai
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.llms.models import Model, model_config


class GridOutput(BaseModel):
    grid: list[list[int]] = Field(..., description="Extracted 2D grid of integers")


async def extract_grid_from_text(
    model: Model,
    text: str,
) -> list[list[int]]:
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"], timeout=120, max_retries=10
    )

    response = await client.chat.completions.create(
        model=model.value,
        messages=[{"role": "user", "content": text}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "extract_grid",
                    "description": "Extract the final 2D integer grid from the given text. The response may contain many 2d integer grids but extract the final answer from the text, which is a 2d list of integers.",
                    "parameters": GridOutput.model_json_schema(),
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "extract_grid"}},
    )

    tool_call = response.choices[0].message.tool_calls[0]
    grid_data = GridOutput.model_validate_json(tool_call.function.arguments)
    return grid_data.grid


async def get_next_message_openai(model: Model, inputs: list[dict[str, str]]) -> str:
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"], timeout=600, max_retries=10
    )
    params = {}
    model_name = model.value

    if model.value.endswith("_high"):
        params["reasoning"] = {"effort": "high"}
        model_name = model.value.replace("_high", "")

    response = await client.responses.create(
        model=model_name,
        max_output_tokens=100_000,
        input=inputs,
        **params,
    )
    # debug(response)
    return response.output_text


async def get_next_message_openrouter(
    model: Model,
    inputs: list[dict[str, str]],
) -> str:
    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=500,
        max_retries=10,
        base_url="https://openrouter.ai/api/v1",
    )
    completion = await client.chat.completions.create(
        model=model.value,
        max_tokens=50_000,
        max_completion_tokens=50_000,
        messages=inputs,
        temperature=1,
    )
    # debug(completion)
    return completion.choices[0].message.content


async def get_next_message_deepseek(
    model: Model,
    inputs: list[dict[str, str]],
) -> str:
    client = AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        timeout=300,
        max_retries=10,
        base_url="https://api.deepseek.com",
    )
    response = await client.chat.completions.create(
        model=model.value,
        max_tokens=8192,
        messages=inputs,
    )
    # debug(response)
    return response.choices[0].message.content


async def get_next_message_anthropic(
    model: Model,
    inputs: list[dict[str, str]],
) -> str:
    config = model_config[model]
    client = AsyncAnthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"], timeout=300, max_retries=30
    )

    new_inputs = []
    for _input in inputs:
        new_inputs.append(
            {
                "role": _input["role"],
                "content": [
                    {
                        "type": "text",
                        "text": _input["content"],
                    }
                ],
            }
        )
    new_inputs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

    params = {}
    if config.max_thinking_tokens:
        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": config.max_thinking_tokens,
        }

    message = await client.messages.create(
        model=model.value,
        max_tokens=config.max_tokens,
        messages=new_inputs,
        **params,
    )
    return message.content[-1].text


async def get_next_message_gemini(
    model: Model,
    inputs: list[dict[str, str]],
) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    contents = []
    for i in inputs:
        role = i["role"]
        if role == "assistant":
            role = "model"
        contents.append(
            genai.types.ContentDict(
                role=role, parts=[genai.types.PartDict(text=i["content"])]
            ),
        )

    response = await client.aio.models.generate_content(
        model=model.value,
        contents=contents,
    )

    return response.text


if __name__ == "__main__":
    import asyncio

    from dotenv import load_dotenv

    load_dotenv()

    asyncio.run(
        get_next_message_openai(
            model="gpt-4.5-preview-2025-02-27",
            inputs=[{"role": "user", "content": "hey how are you?"}],
        )
    )
