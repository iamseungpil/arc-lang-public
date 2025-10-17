# Detailed Prompt Mechanics for ARC v2 Solution

## Overview

My v2 solution evolves natural language instructions instead of Python code, using language models to both generate transformation rules and apply them. This document details the specific prompt engineering techniques that make this approach effective.

## Core Architecture Components

### 1. Instruction Generation Phase

#### Initial Prompt Structure (`INTUITIVE_PROMPT`)
The system uses a carefully crafted prompt that emphasizes pattern recognition:

```
You are participating in a puzzle solving competition. You are an expert at solving puzzles.

Find the common pattern that transforms each input grid into its corresponding output grid...

Your task is to write clear instructions that describe this transformation pattern. These instructions must:
- Apply consistently to ALL training examples
- Be general enough to work on new test cases  
- Be intuitive and easy to understand
- Describe the pattern without referencing specific example numbers
```

**Key Design Decisions:**
- **Role Setting**: Positions the model as an expert puzzle solver to prime pattern recognition capabilities
- **Consistency Emphasis**: Explicitly requires instructions that work for ALL examples, preventing overfitting
- **Generalization Focus**: Instructions must work on unseen test cases, not just training data
- **Intuition Over Technicality**: Prioritizes human-readable explanations over complex algorithmic descriptions

#### Grid Representation
Grids are presented in multiple formats for optimal model comprehension:

1. **ASCII Text Format**: Simple space-separated integers
   ```
   0 0 0 5 0
   0 5 0 0 0
   0 0 0 0 0
   ```

2. **Optional Base64 Images**: Visual representation for models with vision capabilities
   - Generated using matplotlib with consistent color mapping
   - Each integer (0-9) maps to a specific color for visual pattern recognition

3. **Structured Context**: Clear labeling of training examples vs test inputs
   ```
   --Training Examples--
   Training Example 1
   Input:
   [grid]
   Output:
   [grid]
   --End of Training Examples--
   --Test Input--
   [grid]
   ```

### 2. Instruction Following Phase

#### Agent Prompt (`AGENT_FOLLOW_INSTRUCTIONS_PROMPT`)
A separate, focused prompt for applying instructions:

```
You are an expert puzzle solver in a competition.

You will receive:
1. Step-by-step instructions for transforming input grids
2. Training examples showing these instructions applied correctly
3. A test input grid to solve

Your task: Apply the given instructions precisely to transform the test input grid...
```

**Design Rationale:**
- **Separation of Concerns**: Different prompt than generation to avoid confusion
- **Precision Focus**: Emphasizes exact application rather than interpretation
- **Example-Guided**: Training examples serve as implementation reference

#### The "Perfect Instructions" Modifier
When instructions aren't producing perfect results, the system adds:

```
These instructions are a guide to help you get the correct output grid.
If you think there is an error with the instructions that would cause you to get the wrong output, 
ignore that part of the instructions.
What is most important is that you get the exact correct output grid given the general pattern.
```

This allows the model to make small corrections when instructions have minor flaws.

### 3. Revision Mechanisms

#### Individual Revision Prompt (`REVISION_PROMPT`)
```
Your previous instructions were applied to the training input grids, 
but they did not produce the correct output grids.

Below you'll see what outputs were generated when following your instructions. 
Compare these incorrect outputs with the correct outputs...

Based on this feedback, provide updated instructions that:
- Fix the specific errors you observe
- Still work correctly for ALL training examples
- Remain clear, intuitive, and general
```

**Feedback Mechanisms:**

1. **Visual Diff Notation**: ASCII-formatted grid showing cell-by-cell differences
   ```
   +-------+-------+-------+
   |  ✓0   |  2→3  |  ✓5   |  
   +-------+-------+-------+
   |  1→0  |  ✓4   |  ✓5   |
   +-------+-------+-------+
   ```
   - ✓ indicates correct cells
   - actual→expected format for errors
   - Bordered grid for visual clarity

2. **Structured Response Format**:
   ```python
   class ReviseInstructionsResponse:
       reasoning_for_why_old_instructions_are_wrong: str
       revised_instructions: str
   ```
   Forces the model to explicitly analyze failures before proposing fixes.

#### Pooled Revision Strategy
For pooled revisions, the system creates a synthesis prompt showing multiple instruction sets with their scores:

- **Input**: Top 5 instruction sets with their per-example scores
- **Context**: Full training examples with attempts from each instruction set
- **Goal**: Synthesize elements from successful parts of different approaches

**Token Management Challenge**: 
- Thinking models generate extensive reasoning tokens
- Including >2 instructions often exceeds context limits
- System balances information richness vs. practical constraints

### 4. Scoring and Evaluation

#### Leave-One-Out Cross Validation
Each instruction set is tested by:
1. Using it to solve each training example
2. Treating each training example as if it were a test case
3. Calculating cell-wise accuracy for each attempt

#### Grid Similarity Scoring
```python
def get_grid_similarity(ground_truth_grid, sample_grid):
    # Returns percentage of matching cells (0.0 to 1.0)
    # Handles dimension mismatches gracefully
    # Simple but effective for ranking instructions
```

### 5. Model-Specific Optimizations

#### Structured Output Enforcement
Uses function calling/structured generation to ensure valid grid outputs:

```python
class GridResponse(BaseModel):
    grid: list[list[int]] = Field(
        description="The output grid which is the transform instructions applied to the test input"
    )
```

#### Provider-Specific Parameters
- **OpenAI**: Reasoning effort levels ("high" for complex puzzles)
- **Anthropic**: Thinking tokens budget, ephemeral caching
- **Gemini**: Async content generation
- **DeepSeek/Grok**: Extended max_tokens for detailed instructions

### 6. Evolution Strategy Details

#### Population Management
```
Initial: 30 candidates → Filter to top performers
Individual Revision: Top 5 → 5 revised versions  
Pooled Revision: Top 5 → 5 synthesized versions
Total: Maximum 40 instruction attempts per task
```

#### Selection Pressure
- Instructions scoring <0.5 are filtered out early
- Perfect scores (1.0) immediately skip to final application
- Partial scores guide revision focus

### 7. Prompt Engineering Best Practices Applied

1. **Consistent Formatting**: All grids use identical ASCII representation
2. **Progressive Disclosure**: Complex information revealed in stages
3. **Explicit Constraints**: Clear requirements in every prompt
4. **Error Recovery**: Graceful handling of malformed outputs
5. **Context Preservation**: Previous attempts inform revisions
6. **Role Consistency**: Maintain "expert puzzle solver" framing

## Key Insights and Tradeoffs

### Why Natural Language Over Code
- **Expressiveness**: English captures nuanced patterns better than rigid Python
- **Flexibility**: Models can describe transformations that would require complex code
- **Error Tolerance**: Natural language allows for approximate descriptions
- **Model Strengths**: LLMs excel at language manipulation over code generation

### Context Window Management
- **Challenge**: Rich feedback improves results but exceeds token limits
- **Solution**: Adaptive context sizing based on model capabilities
- **Tradeoff**: More examples vs. deeper analysis per example

### Determinism vs. Creativity
- **Structured Outputs**: Ensure valid, parseable results
- **Temperature Control**: Balance between creative solutions and consistency
- **Multiple Attempts**: Diversity through volume rather than randomness

## Implementation Notes

### Async Pipeline Benefits
- Parallel instruction generation and testing
- Concurrent evaluation across training examples  
- Efficient API usage with semaphore control
- Real-time result streaming to database

### Monitoring and Debugging
- Detailed logging of each instruction attempt
- Grid visualization for failed attempts
- Diff notation for quick error analysis
- Database persistence for pattern analysis

## Future Enhancements

1. **Adaptive Prompting**: Adjust prompt complexity based on puzzle difficulty
2. **Meta-Learning**: Use successful instruction patterns to guide future generation
3. **Multi-Modal Fusion**: Better integration of visual and textual pattern recognition
4. **Instruction Compression**: Distill verbose instructions to essential rules
5. **Cross-Model Ensembling**: Combine instructions from different model families

This architecture demonstrates that evolving natural language descriptions can be as powerful as evolving code, while being more interpretable and leveraging the core strengths of modern language models.