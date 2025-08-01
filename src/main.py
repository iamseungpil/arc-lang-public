import os
import traceback
from copy import deepcopy

import logfire
from devtools import debug
from pydantic import BaseModel, Field

# Import logging_config first to apply patches
import src.logging_config  # noqa: F401
from llms.messages import (
    get_next_message_anthropic,
    get_next_message_deepseek,
    get_next_message_gemini,
    get_next_message_openai,
    get_next_message_openrouter,
)
from llms.models import Model
from src.llms.structured import get_next_structure
from src.models import GRID, Challenge, Example, Input


class InstructionsResponse(BaseModel):
    """Given the input / output examples, provide step by step instructions for how to transform the input grids into output grids."""

    instructions: str = Field(
        ...,
        description="Step by step instructions for how to transform the input grids into output grids.",
    )


class ReviseInstructionsResponse(BaseModel):
    """Given the input / output examples and the failed attempts following the previous instructions, provide revised step by step instructions for how to transform the input grids into output grids."""

    reasoning_for_why_old_instructions_are_wrong: str = Field(
        ...,
        description="Give the reasoning for why the old instructions were not correct and how you will improve them to make the revised instructions correct.",
    )
    revised_instructions: str = Field(
        ...,
        description="Revised step by step instructions for how to transform the input grids into output grids taking into account the failed attempts.",
    )


class GridResponse(BaseModel):
    """Using the instructions given, produce the correct output grid from the test input grid."""

    grid: list[list[int]] = Field(
        ...,
        description="The output grid which is the transform instructions applied to the test input grid.",
    )


USE_EXAMPLE_INSTRUCTIONS = False

EXAMPLE_INSTRUCTIONS = """
Here are example instructions that have worked in the past:
<example instructions>
1. Find the background color and ignore that.
2. Take the biggest contiguous shape general shape. This is the outline for the output grid.
3. Fill that largest shape like Tetris with the smaller shapes in ways that they fit together. You will not need to rotate any of the smaller shapes. They each will slot in so that the inner shape is filled and they will also fill in the gaps on the outside of the inner shape.
4. You will be left with a rectangle the height and width of the biggest starting shape.
</example instructions>
"""

EXAMPLE_INSTRUCTIONS_STR = (
    "" if not USE_EXAMPLE_INSTRUCTIONS else f"\n\n{EXAMPLE_INSTRUCTIONS}\n\n"
)

INTUITIVE_PROMPT = f"""
You are participating in a puzzle solving competition. You are an expert at solving puzzles.

Find the common pattern that transforms each input grid into its corresponding output grid, based on the training examples below.

Your task is to write clear instructions that describe this transformation pattern. These instructions must:
- Apply consistently to ALL training examples (the same rule works for every input→output pair)
- Be general enough to work on new test cases
- Be intuitive and easy to understand
- Describe the pattern without referencing specific example numbers or positions

The transformation pattern should be simple and logical - these puzzles are designed to have elegant, intuitive solutions that humans can readily grasp.

Write your instructions as a clear, step-by-step process that someone could follow to transform any input grid into the correct output grid.
{EXAMPLE_INSTRUCTIONS_STR}
Here are the training examples and test input grids:
"""


AGENT_FOLLOW_INSTRUCTIONS_PROMPT = """
You are an expert puzzle solver in a competition.

You will receive:
1. Step-by-step instructions for transforming input grids into output grids
2. Training examples showing these instructions applied correctly
3. A test input grid to solve

Your task: Apply the given instructions precisely to transform the test input grid into its output grid.

The training examples demonstrate how the instructions work - use them to understand the pattern, then follow the exact same process for the test input.
""".strip()

func_to_llm = {
    get_next_message_openai: {
        Model.gpt_4_5,
        Model.o3_mini,
        Model.gpt_4_o,
        Model.o3_mini_high,
        Model.o4_mini_high,
        Model.o4_mini,
    },
    get_next_message_anthropic: {Model.sonnet_3_7, Model.sonnet_3_5},
    get_next_message_gemini: {Model.gemini_2_5},
    get_next_message_deepseek: {Model.deepseek_chat, Model.deepseek_reasoner},
    get_next_message_openrouter: {
        Model.openrouter_sonnet_3_7_thinking,
        Model.openrouter_gemini_2_5_free,
        Model.openrouter_deepseek_3_free,
        Model.openrouter_gemini_2_5,
        Model.openrouter_deepseek_r1,
        Model.openrouter_grok_v3,
        Model.openrouter_quasar_alpha,
        Model.openrouter_sonnet_3_7,
        Model.openrouter_optimus_alpha,
        Model.openrouter_deepseek_r1_free,
    },
}

# now reverse the map
llm_to_func = {}
for f, _llms in func_to_llm.items():
    for _llm in _llms:
        llm_to_func[_llm] = f


class PromptResponse(BaseModel):
    create_instructions: str
    follow_instructions: str


def contents_from_grid(grid: GRID, grid_label: str, include_base64: bool) -> list[dict]:
    contents = [
        {
            "type": "input_text",
            "text": f"{grid_label}:\n{Challenge.grid_to_str(grid=grid)}",
        },
    ]
    if include_base64:
        try:
            contents.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{Challenge.grid_to_base64(grid=grid)}",
                    "detail": "high",
                }
            )
        except (
            ValueError,
            TypeError,
        ) as e:
            logfire.error(f"Error generating base64 image", error=str(e))
            pass

    return contents


def contents_from_example(
    example: Example,
    attempt: GRID | None,
    example_number: int,
    include_base64: bool,
    use_diffs: bool,
) -> list[dict]:
    if not attempt:
        attempt_messages = []
    else:
        if attempt != example.output:
            failed_str = "Failed Output Attempt"
        else:
            failed_str = "Successful Output Attempt"
        attempt_messages = contents_from_grid(
            grid=attempt, grid_label=failed_str, include_base64=include_base64
        )

        # Add diff notation if use_diffs is True and attempt failed
        if use_diffs and attempt != example.output:
            from src.run import generate_grid_diff

            diff_text = generate_grid_diff(
                expected_grid=example.output, actual_grid=attempt
            )
            attempt_messages.append(
                {
                    "type": "input_text",
                    "text": f"\nDifference notation (actual→expected):\n{diff_text}",
                }
            )

    return [
        {"type": "input_text", "text": f"Training Example {example_number}\n"},
        *contents_from_grid(
            grid=example.input, grid_label="Input", include_base64=include_base64
        ),
        *contents_from_grid(
            grid=example.output, grid_label="Output", include_base64=include_base64
        ),
        *attempt_messages,
    ]


def contents_from_challenge(
    training_examples: list[Example],
    training_example_attempts: list[GRID] | None,
    test_inputs: list[Input],
    include_base64: bool,
    use_diffs: bool,
) -> list[dict]:
    contents: list[dict] = [{"type": "input_text", "text": "--Training Examples--"}]
    for i, example in enumerate(training_examples):
        attempt = (
            None if not training_example_attempts else training_example_attempts[i]
        )
        contents.extend(
            contents_from_example(
                example=example,
                attempt=attempt,
                example_number=i + 1,
                include_base64=include_base64,
                use_diffs=use_diffs,
            )
        )
    if len(test_inputs) == 1:
        test_inputs_str = "Test Input"
    else:
        test_inputs_str = "Test Inputs"
    contents.append(
        {
            "type": "input_text",
            "text": f"--End of Training Examples--\n\n--{test_inputs_str}--",
        }
    )
    for i, test in enumerate(test_inputs):
        contents.extend(
            contents_from_grid(
                grid=test.input,
                grid_label=f"Test Input {i + 1}",
                include_base64=include_base64,
            )
        )
    contents.append({"type": "input_text", "text": f"--End of {test_inputs_str}--"})
    return contents


PERFECT_PROMPT = """
These instructions are a guide to help you get the correct output grid.
If you think there is an error with the instructions that would cause you to get the wrong output grid, ignore that part of the instructions.
What is most important is that you get the exact correct output grid given the general pattern you observe.
""".strip()


async def output_grid_from_instructions(
    instructions: str,
    training_examples: list[Example],
    test_input_grid: GRID,
    model: Model,
    include_base64: bool,
    use_diffs: bool,
    is_perfect: bool,
) -> GRID:
    contents_from_examples: list[dict] = []
    for i, example in enumerate(training_examples):
        contents_from_examples.extend(
            contents_from_example(
                example=example,
                example_number=i + 1,
                include_base64=include_base64,
                attempt=None,
                use_diffs=use_diffs,
            )
        )
    perfect_messages = []
    if not is_perfect:
        perfect_messages = [{"type": "input_text", "text": PERFECT_PROMPT}]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": AGENT_FOLLOW_INSTRUCTIONS_PROMPT,
                },
                {"type": "input_text", "text": f"\nInstructions:\n{instructions}"},
                *perfect_messages,
                *contents_from_examples,
                *contents_from_grid(
                    grid=test_input_grid,
                    grid_label="Test Input Grid",
                    include_base64=include_base64,
                ),
                {"type": "input_text", "text": "Test Output Grid:"},
            ],
        }
    ]
    return (
        await get_next_structure(structure=GridResponse, messages=messages, model=model)
    ).grid


def prompt_from_challenge(c: Challenge) -> list[PromptResponse]:
    # TODO later: add a reasoning sentence to the prompt... this requires some thought. For now can use reasoning models
    example_strs: list[str] = []
    for i, example in enumerate(c.train):
        _input_str = c.grid_to_str(grid=example.input)
        output_str = c.grid_to_str(grid=example.output)
        example_strs.append(
            f"Example {i + 1}:\n\nInput:\n{_input_str}\nOutput:\n{output_str}"
        )
    example_str = "\n\n".join(example_strs)
    responses: list[PromptResponse] = []
    for test_example in c.test:
        responses.append(
            PromptResponse(
                create_instructions=f"""
{INTUITIVE_PROMPT}

{example_str}
            """.strip(),
                follow_instructions=f"Apply the instructions you created to create the output grid from this input grid:\n{c.grid_to_str(grid=test_example.input)}",
            )
        )
    return responses


class ChallengeError(Exception):
    pass


async def _solve_challenge(c: Challenge, model: Model) -> list[GRID]:
    logfire.debug("Starting challenge solve", model=model.value)
    get_next_f = llm_to_func[model]
    prompts = prompt_from_challenge(c=c)
    responses: list[GRID] = []
    for prompt in prompts:
        logfire.debug("Processing prompt", prompt=prompt.model_dump())
        inputs = [{"role": "user", "content": prompt.create_instructions}]
        params = {"model": model, "inputs": deepcopy(inputs)}
        response = await get_next_f(**params)
        logfire.debug("First LLM response", response=response)
        inputs.extend(
            [
                {"role": "assistant", "content": response},
                {"role": "user", "content": prompt.follow_instructions},
            ]
        )
        params["inputs"] = inputs
        final_response = await get_next_f(**params)
        logfire.debug("Final LLM response", response=final_response)
        try:
            parsed_grid = await Challenge.grid_from_str_using_llm(s=final_response)
            logfire.debug("Grid parsed successfully", grid=parsed_grid)
            responses.append(parsed_grid)
        except Exception as e:
            logfire.error("Error parsing grid", error=str(e))
            raise ChallengeError(e)
    return responses


async def _solve_challenge_times(
    c: Challenge, model: Model, times: int
) -> list[list[GRID]]:
    responses = await asyncio.gather(
        *[_solve_challenge(c=c, model=model) for _ in range(times)],
        return_exceptions=True,
    )
    exceptions = [r for r in responses if isinstance(r, Exception)]
    if exceptions:
        logfire.error(
            "Exceptions in solve_challenge_times", exception_count=len(exceptions)
        )
        for e in exceptions:
            logfire.error(
                f"Exception: {type(e).__name__}",
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc(),
            )

    return [r for r in responses if not isinstance(r, Exception)]


async def solve_challenge(
    challenge_id: str,
    challenge: Challenge,
    solutions: list[GRID],
    model: Model,
    times: int,
) -> None:
    # get challenge
    # create prompt from challenge
    # send prompt to llm
    # test llm answer with other llms

    correct_count = 0
    total_count = 0
    answers_list = await _solve_challenge_times(c=challenge, model=model, times=times)
    for answers in answers_list:
        for answer_grid, solution in zip(answers, solutions, strict=True):
            is_correct = answer_grid == solution
            debug(is_correct)
            if is_correct:
                correct_count += 1
            total_count += 1
        if os.environ.get("VIZ", "0") == "1":
            challenge.viz(
                # train_attempts=[[] for _ in range(len(challenge.train))],
                solutions=solutions,
                test_attempts=answers,
            )

    pct = correct_count / total_count if total_count > 0 else 0

    message = f"[{challenge_id}], [{model.value}]: got {correct_count} correct out of {total_count}, times: {times}, which is {pct * 100:.2f}%"

    with open("results.txt", "a+") as ff:
        ff.write(message + "\n")

    logfire.info(
        "Challenge results",
        challenge_id=challenge_id,
        model=model.value,
        correct_count=correct_count,
        total_count=total_count,
        times=times,
        success_rate=f"{pct * 100:.2f}%",
    )


async def run(
    challenge_ids: list[str],
    challenges_by_id: dict[str, Challenge],
    solutions_by_id: dict[str, list[list[list[int]]]],
    model: Model,
    times: int,
    max_challenges_in_parallel: int,
) -> None:
    # Create a semaphore to limit parallel execution
    semaphore = asyncio.Semaphore(max_challenges_in_parallel)

    async def solve_with_semaphore(challenge_id: str) -> None:
        async with semaphore:
            await solve_challenge(
                challenge_id=challenge_id,
                challenge=challenges_by_id[challenge_id],
                solutions=solutions_by_id[challenge_id],
                model=model,
                times=times,
            )

    # Create tasks with semaphore control
    tasks = [solve_with_semaphore(challenge_id) for challenge_id in challenge_ids]

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)


def challenge_ids_by_size(challenges_by_id: dict[str, Challenge]) -> list[str]:
    # sort challenges by size and return list of challenge ids
    return sorted(challenges_by_id.keys(), key=lambda k: challenges_by_id[k].size())


if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    from pydantic import TypeAdapter

    challenges_dir = (
        Path(__file__).parent.parent
        / "data"
        / "arc-prize-2025"
        / "arc-agi_evaluation_challenges.json"
    )
    root_challenges = TypeAdapter(dict[str, Challenge]).validate_json(
        challenges_dir.read_text()
    )

    solutions_dir = (
        Path(__file__).parent.parent
        / "data"
        / "arc-prize-2025"
        / "arc-agi_evaluation_solutions.json"
    )
    root_solutions = TypeAdapter(dict[str, list[list[list[int]]]]).validate_json(
        solutions_dir.read_text()
    )

    # challenge_ids = list(root_challenges.keys())[0:5]
    challenge_ids = challenge_ids_by_size(challenges_by_id=root_challenges)[0:5]

    # challenge_ids = ["007bbfb7"]
    # challenge_ids = ["007bbfb7", "00d62c1b", ]
    # challenge_ids = ["045e512c"]
    # challenge_ids = ["06df4c85"]
    # challenge_ids = ["00576224"]
    # challenge_ids = ["184a9768"]
    # challenge_ids = ["e619ca6e"]
    asyncio.run(
        run(
            challenge_ids=challenge_ids,
            challenges_by_id=root_challenges,
            solutions_by_id=root_solutions,
            model=Model.o4_mini,
            times=50,
            max_challenges_in_parallel=2,
        )
    )
