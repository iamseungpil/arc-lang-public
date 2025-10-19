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
    headers = dict(RESPONSES_EXTRA_HEADERS)
    headers.update(extra_headers)

    is_gpt5_model = model in {Model.gpt_5, Model.gpt_5_pro}

    # Configure defaults for GPT-5 models
    if is_gpt5_model:
        # Default to streaming for better UX
        if "stream" not in create_kwargs:
            create_kwargs["stream"] = True

        # Enable background mode for GPT-5 models - REQUIRED for long-running tasks
        # GPT-5 models can think for over an hour, exceeding foreground 1h TTL
        # Background mode allows unlimited runtime with proper timeout handling
        # Note: There's a known latency issue - time to first token is higher with background mode
        # but this is necessary to prevent timeouts on long reasoning tasks
        if "background" not in extra_body:
            extra_body["background"] = True

        # Background mode requires store=True
        create_kwargs["store"] = True

        # Set verbosity to low for GPT-5-Pro to reduce token usage
        # Verbosity is nested inside the 'text' parameter
        if model == Model.gpt_5_pro and "text" not in create_kwargs:
            create_kwargs["text"] = {"verbosity": "low"}
        elif model == Model.gpt_5_pro and "text" in create_kwargs:
            # If text already exists, merge verbosity into it
            if isinstance(create_kwargs["text"], dict) and "verbosity" not in create_kwargs["text"]:
                create_kwargs["text"]["verbosity"] = "low"
    else:
        # For non-GPT-5 models, store responses by default
        if "store" not in create_kwargs:
            create_kwargs["store"] = True

    if extra_body:
        create_kwargs["extra_body"] = extra_body

    streaming_requested = bool(create_kwargs.get("stream"))
    is_background_mode = bool(extra_body.get("background", False))
    verbosity = None
    if isinstance(create_kwargs.get("text"), dict):
        verbosity = create_kwargs["text"].get("verbosity")

    log.info(
        "openai_request_config",
        model=model.value,
        streaming=streaming_requested,
        background=is_background_mode,
        store=create_kwargs.get("store"),
        verbosity=verbosity,
    )

    if streaming_requested:
        return await _handle_streaming_response(
            client=client,
            model=model,
            create_kwargs=create_kwargs,
            headers=headers,
            is_background_mode=is_background_mode,
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
    is_background_mode: bool,
) -> Response:
    """Handle streaming responses with support for background mode."""
    # Enable detailed reasoning summaries for GPT-5 models
    if model in {Model.gpt_5, Model.gpt_5_pro} and "reasoning" not in create_kwargs:
        create_kwargs["reasoning"] = {"generate_summary": "detailed"}

    # When using background + stream together, responses.create() returns a stream directly
    # Remove stream from kwargs for the standard (non-background) path only
    if not is_background_mode:
        create_kwargs.pop("stream", None)
        stream_manager = client.responses.stream(extra_headers=headers, **create_kwargs)
    else:
        # For background streaming, create() with stream=True returns AsyncStream directly
        stream_manager = await client.responses.create(extra_headers=headers, **create_kwargs)

    final_response: Response | None = None
    response_id: str | None = None
    last_sequence_number: int | None = None

    async with stream_manager as stream:
        async for event in stream:
            _log_stream_event(model, event)

            # Track sequence_number for resumable background streams
            sequence_number = getattr(event, "sequence_number", None)
            if sequence_number is not None:
                last_sequence_number = sequence_number

            event_type = getattr(event, "type", None)
            if event_type == "response.created" and response_id is None:
                response = getattr(event, "response", None)
                response_id = getattr(response, "id", None)
            elif event_type in {"response.completed", "response.failed", "response.incomplete"}:
                final_response = getattr(event, "response", None)

    # Log cursor for resumable background streams
    if is_background_mode and last_sequence_number is not None:
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
    response = await client.responses.create(extra_headers=headers, **create_kwargs)

    status = response.status or "completed"
    response_id = response.id

    # Return immediately if already completed
    if status in POLL_TERMINAL_STATUSES:
        return response

    # Poll for completion with exponential backoff
    poll_interval = POLL_DEFAULT_INTERVAL
    start_time = time.time()
    current = response

    while status not in POLL_TERMINAL_STATUSES:
        if not response_id:
            raise RuntimeError(
                f"Response missing ID while status={status} for {model.value}"
            )

        # Check timeout (3 hours for long-running background tasks)
        if time.time() - start_time > POLL_TIMEOUT_SECONDS:
            raise TimeoutError(
                f"Response polling exceeded {POLL_TIMEOUT_SECONDS}s timeout for {model.value}"
            )

        await asyncio.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.5, POLL_MAX_INTERVAL)

        current = await client.responses.retrieve(response_id, extra_headers=headers)
        status = current.status or "completed"

    # Check for error statuses
    if status in POLL_ERROR_STATUSES:
        error_detail = current.error or current.model_dump()
        raise RuntimeError(
            f"Response failed with status={status} for {model.value}: {error_detail}"
        )
    if status == "requires_action":
        raise RuntimeError(
            f"Response requires action (tool calls not supported) for {model.value}"
        )

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
        # Regular output text streaming
        log.debug(
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
