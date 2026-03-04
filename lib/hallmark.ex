defmodule Hallmark do
  @moduledoc """
  HHEM (Hallucination Evaluation Model) for Elixir.

  Scores (premise, hypothesis) pairs from 0 (hallucinated) to 1 (consistent)
  using Vectara's HHEM model, a fine-tuned FLAN-T5-base.
  """

  alias Hallmark.Model

  @default_threshold 0.5

  @doc """
  Loads the HHEM model, tokenizer, and classifier weights.

  Downloads from HuggingFace on first call, then uses cached files.

  ## Options

    * `:compiler` - Nx compiler for inference (e.g. `EXLA`). Without one, uses the
      default Nx.Defn evaluator which is very slow for transformer models.

    * `:max_length` - Maximum token sequence length (default: `2048`). Increase if
      premises exceed 2048 tokens; the underlying T5 model supports arbitrary lengths
      via relative position biases.

  ## Examples

      {:ok, model} = Hallmark.load(compiler: EXLA)
  """
  def load(opts \\ []) do
    Model.load(opts)
  end

  @doc """
  Scores a single (premise, hypothesis) pair.

  Returns a float from 0.0 (hallucinated) to 1.0 (consistent).

  ## Examples

      {:ok, score} = Hallmark.score(model, "I am in California", "I am in United States.")
  """
  def score(%Model{} = model, premise, hypothesis) do
    Model.predict(model, premise, hypothesis)
  end

  @doc """
  Scores a batch of (premise, hypothesis) pairs.

  ## Examples

      {:ok, scores} = Hallmark.score_batch(model, [
        {"I am in California", "I am in United States."},
        {"The capital of France is Berlin.", "The capital of France is Paris."}
      ])
  """
  def score_batch(%Model{} = model, pairs) do
    Model.predict_batch(model, pairs)
  end

  @doc """
  Evaluates a pair and returns `:consistent` or `:hallucinated`.

  ## Options

    * `:threshold` - Score threshold (default: #{@default_threshold}).
      Scores >= threshold are `:consistent`, below are `:hallucinated`.

  ## Examples

      {:ok, :consistent} = Hallmark.evaluate(model, "I am in California", "I am in United States.")
      {:ok, :hallucinated} = Hallmark.evaluate(model, "The capital of France is Berlin.", "The capital of France is Paris.")
  """
  def evaluate(%Model{} = model, premise, hypothesis, opts \\ []) do
    threshold = Keyword.get(opts, :threshold, @default_threshold)

    with {:ok, score} <- score(model, premise, hypothesis) do
      label = if score >= threshold, do: :consistent, else: :hallucinated
      {:ok, label}
    end
  end
end
