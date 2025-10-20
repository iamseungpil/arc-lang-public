import asyncio
import json
import time
import typing as T

from openai import AsyncOpenAI
from openai.types.responses.response import Response

from src.llms.models import Model
from src.log import log
from openai.types.responses.response_stream_event import ResponseStreamEvent


RESPONSES_EXTRA_HEADERS = {"OpenAI-Beta": "responses=v2"}
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
POLL_TIMEOUT_SECONDS = 10_800.0

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
    """
    Create and handle OpenAI Response API requests with support for streaming and background mode.

    For GPT-5 models:
    - Defaults to background mode with streaming for reliability and resumability
    - Sets verbosity to 'low' for GPT-5-Pro to reduce token usage
    - Enables detailed reasoning summaries when streaming
    - Enforces store=True (required for background mode)
    """
    create_kwargs = dict(create_kwargs)
    extra_body: dict[str, T.Any] = dict(create_kwargs.pop("extra_body", {}) or {})
    extra_headers: dict[str, T.Any] = dict(create_kwargs.pop("extra_headers", {}) or {})
    headers = {**RESPONSES_EXTRA_HEADERS, **extra_headers}

    is_gpt5 = model in {Model.gpt_5, Model.gpt_5_pro}

    # Configure GPT-5 defaults: streaming, background mode, low verbosity for Pro
    if is_gpt5:
        create_kwargs.setdefault("stream", True)
        create_kwargs["store"] = True
        extra_body.setdefault("background", True)

        # Set low verbosity for GPT-5-Pro to reduce token usage
        if model == Model.gpt_5_pro:
            text_config = create_kwargs.setdefault("text", {})
            if isinstance(text_config, dict):
                text_config.setdefault("verbosity", "low")
    else:
        create_kwargs.setdefault("store", True)

    if extra_body:
        create_kwargs["extra_body"] = extra_body

    log.info(
        "openai_request_config",
        model=model.value,
        streaming=create_kwargs.get("stream", False),
        background=extra_body.get("background", False),
        store=create_kwargs.get("store"),
        verbosity=create_kwargs.get("text", {}).get("verbosity") if isinstance(create_kwargs.get("text"), dict) else None,
        tools=len(create_kwargs.get("tools", [])),
    )

    if create_kwargs.get("stream"):
        return await _handle_streaming_response(
            client=client,
            model=model,
            create_kwargs=create_kwargs,
            headers=headers,
        )
    else:
        return await _handle_polling_response(
            client=client,
            model=model,
            create_kwargs=create_kwargs,
            headers=headers,
        )


async def _handle_streaming_response(
    client: AsyncOpenAI,
    model: Model,
    create_kwargs: dict[str, T.Any],
    headers: dict[str, T.Any],
) -> Response:
    """Handle streaming responses with support for background mode."""
    from openai import APIError

    # Enable detailed reasoning summaries for GPT-5 models
    if model in {Model.gpt_5, Model.gpt_5_pro} and "reasoning" not in create_kwargs:
        create_kwargs["reasoning"] = {"generate_summary": "detailed"}

    # GPT-5 models always use background streaming (create returns AsyncStream directly)
    # Other models use standard streaming
    is_gpt5 = model in {Model.gpt_5, Model.gpt_5_pro}

    final_response: Response | None = None
    response_id: str | None = None
    last_sequence_number: int | None = None
    max_retries = 3
    retry_delay = 2.0

    for attempt in range(max_retries):
        try:
            # Create stream manager
            if is_gpt5:
                # Background streaming for GPT-5: create() returns stream directly
                stream_manager = await client.responses.create(extra_headers=headers, **create_kwargs)
            else:
                # Standard streaming for other models
                create_kwargs.pop("stream", None)
                stream_manager = client.responses.stream(extra_headers=headers, **create_kwargs)

            async with stream_manager as stream:
                async for event in stream:
                    _log_stream_event(model, event)

                    # Track sequence_number for resumable streams
                    sequence_number = getattr(event, "sequence_number", None)
                    if sequence_number is not None:
                        last_sequence_number = sequence_number

                    event_type = getattr(event, "type", None)
                    if event_type == "response.created" and response_id is None:
                        response = getattr(event, "response", None)
                        response_id = getattr(response, "id", None)
                    elif event_type in {"response.completed", "response.failed", "response.incomplete"}:
                        final_response = getattr(event, "response", None)
            break  # Success
        except APIError as e:
            if attempt < max_retries - 1:
                log.warning(
                    "openai_stream_retry",
                    model=model.value,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                )
                await asyncio.sleep(retry_delay * (attempt + 1))
            else:
                log.error(
                    "openai_stream_error",
                    model=model.value,
                    response_id=response_id,
                    error=str(e),
                )
                raise

    # Log final cursor for resumable streams
    if is_gpt5 and last_sequence_number is not None:
        log.info(
            "openai_stream_complete",
            model=model.value,
            response_id=response_id,
            last_sequence_number=last_sequence_number,
        )

    # Return final response or retrieve it
    if final_response is not None:
        return final_response
    if response_id:
        return await client.responses.retrieve(response_id, extra_headers=headers)
    raise RuntimeError(f"Streaming response finished without terminal event for {model.value}")


async def _handle_polling_response(
    client: AsyncOpenAI,
    model: Model,
    create_kwargs: dict[str, T.Any],
    headers: dict[str, T.Any],
) -> Response:
    """Handle non-streaming responses with polling for background mode."""
    current = await client.responses.create(extra_headers=headers, **create_kwargs)
    status = current.status or "completed"

    if status in POLL_TERMINAL_STATUSES:
        return current

    # Poll with exponential backoff (max 3 hours for long-running tasks)
    response_id = current.id
    if not response_id:
        raise RuntimeError(f"Response missing ID for {model.value}")

    poll_interval = POLL_DEFAULT_INTERVAL
    start_time = time.time()

    while status not in POLL_TERMINAL_STATUSES:
        if time.time() - start_time > POLL_TIMEOUT_SECONDS:
            raise TimeoutError(f"Response polling timeout after {POLL_TIMEOUT_SECONDS}s for {model.value}")

        await asyncio.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.5, POLL_MAX_INTERVAL)
        current = await client.responses.retrieve(response_id, extra_headers=headers)
        status = current.status or "completed"

    # Check final status
    if status in POLL_ERROR_STATUSES:
        raise RuntimeError(f"Response {status} for {model.value}: {current.error or current.model_dump()}")
    if status == "requires_action":
        raise RuntimeError(f"Response requires action (not supported) for {model.value}")

    return current


def _log_stream_event(model: Model, event: ResponseStreamEvent) -> None:
    """Log streaming events including reasoning summaries."""
    event_type = getattr(event, "type", "unknown")
    sequence_number = getattr(event, "sequence_number", None)

    # Background mode status events
    if event_type == "response.queued":
        response = getattr(event, "response", None)
        log.info(
            "openai_response_queued",
            model=model.value,
            response_id=getattr(response, "id", None),
            status=getattr(response, "status", None),
            sequence_number=sequence_number,
        )
    elif event_type == "response.in_progress":
        response = getattr(event, "response", None)
        log.info(
            "openai_response_in_progress",
            model=model.value,
            response_id=getattr(response, "id", None),
            status=getattr(response, "status", None),
            sequence_number=sequence_number,
        )
    # Content streaming events
    elif event_type == "response.output_text.delta":
        # Regular output text streaming - use INFO so it shows in Logfire real-time
        log.info(
            "openai_stream_text_delta",
            model=model.value,
            delta=getattr(event, "delta", ""),
            output_index=getattr(event, "output_index", None),
            content_index=getattr(event, "content_index", None),
            sequence_number=sequence_number,
        )
    elif event_type == "response.output_item.added":
        # New output item started
        item = getattr(event, "item", None)
        item_type = getattr(item, "type", None) if item else None
        log.info(
            "openai_output_item_added",
            model=model.value,
            output_index=getattr(event, "output_index", None),
            item_type=item_type,
            sequence_number=sequence_number,
        )
    # Reasoning events
    elif event_type == "response.reasoning_summary_text.delta":
        # Reasoning summary streaming in real-time
        log.info(
            "openai_reasoning_summary_delta",
            model=model.value,
            delta=getattr(event, "delta", ""),
            output_index=getattr(event, "output_index", None),
            sequence_number=sequence_number,
        )
    elif event_type == "response.reasoning_summary_text.added":
        # Reasoning summary started
        log.info(
            "openai_reasoning_summary_started",
            model=model.value,
            output_index=getattr(event, "output_index", None),
            sequence_number=sequence_number,
        )
    elif event_type == "response.reasoning_summary_text.done":
        # Reasoning summary completed
        log.info(
            "openai_reasoning_summary_done",
            model=model.value,
            output_index=getattr(event, "output_index", None),
            sequence_number=sequence_number,
        )
    # Lifecycle events
    elif event_type == "response.created":
        response = getattr(event, "response", None)
        log.info(
            "openai_stream_started",
            model=model.value,
            response_id=getattr(response, "id", None),
        )
    elif event_type == "response.completed":
        response = getattr(event, "response", None)
        log.info(
            "openai_stream_completed",
            model=model.value,
            response_id=getattr(response, "id", None),
        )
    elif event_type == "response.failed":
        response = getattr(event, "response", None)
        log.error(
            "openai_stream_failed",
            model=model.value,
            response_id=getattr(response, "id", None),
            error=getattr(event, "error", None),
        )
    else:
        # Log all other events for debugging
        log.debug(
            "openai_stream_event",
            model=model.value,
            event_type=event_type,
            sequence_number=sequence_number,
        )


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
