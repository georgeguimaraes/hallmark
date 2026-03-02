# Humble

Hallucination detection for Elixir, powered by Vectara's [HHEM](https://huggingface.co/vectara/hallucination_evaluation_model) (Hallucination Evaluation Model).

Given a premise and a hypothesis, Humble scores how consistent the hypothesis is with the premise on a scale from 0 (hallucinated) to 1 (consistent). Useful for checking whether LLM-generated text is actually grounded in the source material.

HHEM is a fine-tuned FLAN-T5-base (184M params) that runs locally via Bumblebee. No API keys, no external calls after the initial model download.

## Installation

```elixir
def deps do
  [
    {:humble, "~> 0.1.0"},
    {:exla, "~> 0.9"}
  ]
end
```

EXLA is technically optional but you really want it. Without it, inference falls back to the pure Elixir evaluator which takes minutes per prediction instead of seconds.

## Usage

```elixir
# Load the model (downloads ~440MB on first run, cached after that)
{:ok, model} = Humble.load(compiler: EXLA)

# Score a single pair
{:ok, score} = Humble.score(model, "I am in California", "I am in United States.")
# => {:ok, 0.65}

# Hallucination detected
{:ok, score} = Humble.score(model, "The capital of France is Berlin.", "The capital of France is Paris.")
# => {:ok, 0.01}

# Get a label instead of a score
{:ok, :consistent} = Humble.evaluate(model, "I am in California", "I am in United States.")
{:ok, :hallucinated} = Humble.evaluate(model, "The capital of France is Berlin.", "The capital of France is Paris.")

# Custom threshold (default is 0.5)
{:ok, label} = Humble.evaluate(model, premise, hypothesis, threshold: 0.8)

# Batch scoring
{:ok, scores} = Humble.score_batch(model, [
  {"I am in California", "I am in United States."},
  {"I am in United States", "I am in California."},
  {"Mark Wahlberg was a fan of Manny.", "Manny was a fan of Mark Wahlberg."}
])
```

## How it works

HHEM checks logical entailment, not factual accuracy. It answers "does the hypothesis follow from the premise?" rather than "is the hypothesis true?" So a factually correct statement can still be flagged as hallucinated if it doesn't follow from the given premise.

Under the hood, Humble loads the T5 encoder from HHEM's fine-tuned weights, runs the input through the encoder, and applies a 2-class classifier head on the pad token embedding. The softmax probability of the "consistent" class is the score.

## License

MIT
