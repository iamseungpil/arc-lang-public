import asyncio
import json
import os
import time
import typing as T

from openai import AsyncOpenAI
from openai.types.responses.response import Response

from src.llms.models import Model


RESPONSES_EXTRA_HEADERS = {"OpenAI-Beta": "responses=v2"}
RESPONSES_STORE_DISABLED = os.getenv("OPENAI_RESPONSES_STORE", "0") not in {"1", "true", "TRUE"}
POLL_TERMINAL_STATUSES = {
    "completed",
    "cancelled",
    "failed",
    "expired",
    "succeeded",
}
POLL_ERROR_STATUSES = {
    "cancelled",
    "failed",
    "expired",
    "rejected",
}
POLL_DEFAULT_INTERVAL = 2.0
POLL_MAX_INTERVAL = 15.0
POLL_TIMEOUT_SECONDS = 3_600.0

OPENAI_MODEL_MAX_OUTPUT_TOKENS: dict[Model, int] = {
    Model.gpt_4_5: 100_000,
    Model.gpt_4_o: 100_000,
    Model.o3_mini: 100_000,
    Model.o3_mini_high: 100_000,
    Model.o4_mini_high: 128_000,
    Model.o3: 128_000,
    Model.o3_pro: 128_000,
    Model.o4_mini: 128_000,
    Model.gpt_4_1: 128_000,
    Model.gpt_4_1_mini: 128_000,
    Model.gpt_5: 128_000,
    Model.gpt_5_pro: 200_000,
}


async def create_and_poll_response(
    client: AsyncOpenAI,
    *,
    model: Model,
    create_kwargs: dict[str, T.Any],
) -> Response:
    if RESPONSES_STORE_DISABLED and "store" not in create_kwargs:
        create_kwargs["store"] = False

    response = await client.responses.create(
        extra_headers=RESPONSES_EXTRA_HEADERS,
        **create_kwargs,
    )

    status = response.status or "completed"
    response_id = response.id
    poll_interval = POLL_DEFAULT_INTERVAL
    start_time = time.time()

    current = response
    while status not in POLL_TERMINAL_STATUSES:
        if not response_id:
            raise RuntimeError(
                f"OpenAI response missing id while status={status} for model={model.value}"
            )
        await asyncio.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.5, POLL_MAX_INTERVAL)
        current = await client.responses.retrieve(
            response_id,
            extra_headers=RESPONSES_EXTRA_HEADERS,
        )
        status = current.status or "completed"
        if time.time() - start_time > POLL_TIMEOUT_SECONDS:
            raise TimeoutError(
                f"Polling OpenAI response exceeded timeout for model={model.value}"
            )

    if status in POLL_ERROR_STATUSES:
        error_detail = current.error or current.model_dump()
        raise RuntimeError(
            f"OpenAI response failed with status={status} for model={model.value}: {error_detail}"
        )
    if status == "requires_action":
        raise RuntimeError(
            f"OpenAI response requires action (tool calls not supported) for model={model.value}"
        )
    return current


def extract_structured_output(response: Response | dict[str, T.Any]) -> dict[str, T.Any]:
    """
    Extract the structured JSON payload emitted by the Responses API when using json_schema format.
    """

    if isinstance(response, Response):
        payload = response.model_dump()
    else:
        payload = response

    outputs = payload.get("output") or []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        contents = item.get("content") or []
        for content in contents:
            if not isinstance(content, dict):
                continue
            if "json" in content and isinstance(content["json"], dict):
                return content["json"]
            text = content.get("text")
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    continue

    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        try:
            return json.loads(output_text)
        except json.JSONDecodeError:
            pass

    raise ValueError("Unable to extract structured JSON output from OpenAI response.")
