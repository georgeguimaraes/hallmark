# Hallmark

[![Run in Livebook](https://livebook.dev/badge/v1/blue.svg)](https://livebook.dev/run?url=https%3A%2F%2Fgithub.com%2Fgeorgeguimaraes%2Fhallmark%2Fblob%2Fmain%2Flivebooks%2Fhallmark_guide.livemd)

Hallucination detection for Elixir, powered by Vectara's [HHEM](https://huggingface.co/vectara/hallucination_evaluation_model) (Hallucination Evaluation Model).

Given a premise and a hypothesis, Hallmark scores how consistent the hypothesis is with the premise on a scale from 0 (hallucinated) to 1 (consistent). Useful for checking whether LLM-generated text is actually grounded in the source material.

HHEM is a fine-tuned FLAN-T5-base (184M params) that runs locally via Bumblebee. No API keys, no external calls after the initial model download.

## Installation

```elixir
def deps do
  [
    {:hallmark, "~> 0.1.0"},
    {:exla, "~> 0.9"}
  ]
end
```

You need a compiler like EXLA or EMLX. Without one, the pure Elixir evaluator runs each tensor op individually and a single score call takes 10+ minutes. With EXLA, it's ~170ms.

EXLA works on all platforms (CPU on Mac, CUDA on Linux with a GPU). If you're on Apple Silicon and want Metal GPU acceleration, use [EMLX](https://github.com/elixir-nx/emlx) instead:

```elixir
{:emlx, "~> 0.2"}
```

## Usage

```elixir
# Load the model (downloads ~440MB on first run, cached after that)
{:ok, model} = Hallmark.load(compiler: EXLA)

# Score a single pair
{:ok, score} = Hallmark.score(model, "I am in California", "I am in United States.")
# => {:ok, 0.65}

# Hallucination detected
{:ok, score} = Hallmark.score(model, "The capital of France is Berlin.", "The capital of France is Paris.")
# => {:ok, 0.01}

# Get a label instead of a score
{:ok, :consistent} = Hallmark.evaluate(model, "I am in California", "I am in United States.")
{:ok, :hallucinated} = Hallmark.evaluate(model, "The capital of France is Berlin.", "The capital of France is Paris.")

# Custom threshold (default is 0.5)
{:ok, label} = Hallmark.evaluate(model, premise, hypothesis, threshold: 0.8)

# Batch scoring
{:ok, scores} = Hallmark.score_batch(model, [
  {"I am in California", "I am in United States."},
  {"I am in United States", "I am in California."},
  {"Mark Wahlberg was a fan of Manny.", "Manny was a fan of Mark Wahlberg."}
])
```

## How it works

HHEM checks logical entailment, not factual accuracy. It answers "does the hypothesis follow from the premise?" rather than "is the hypothesis true?" So a factually correct statement can still be flagged as hallucinated if it doesn't follow from the given premise.

Under the hood, Hallmark loads the T5 encoder from HHEM's fine-tuned weights, runs the input through the encoder, and applies a 2-class classifier head on the pad token embedding. The softmax probability of the "consistent" class is the score.

## License

MIT
