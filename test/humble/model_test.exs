defmodule Humble.ModelTest do
  use ExUnit.Case

  @moduletag :slow

  @test_pairs [
    {"The capital of France is Berlin.", "The capital of France is Paris.", 0.01},
    {"I am in California", "I am in United States.", 0.65},
    {"I am in United States", "I am in California.", 0.13},
    {"A person on a horse jumps over a broken down airplane.",
     "A person is outdoors, on a horse.", nil},
    {"A boy is jumping on skateboard in the middle of a red bridge.",
     "The boy skates down the sidewalk on a red bridge", nil},
    {"A man with blond-hair, and a brown shirt drinking out of a public water fountain.",
     "A blond man wearing a brown shirt is reading a book.", nil},
    {"Mark Wahlberg was a fan of Manny.", "Manny was a fan of Mark Wahlberg.", nil}
  ]

  setup_all do
    {:ok, model} = Humble.load(compiler: EXLA)
    %{model: model}
  end

  describe "score/3" do
    test "scores known pairs within tolerance", %{model: model} do
      for {premise, hypothesis, expected} <- @test_pairs, expected != nil do
        {:ok, score} = Humble.score(model, premise, hypothesis)

        assert_in_delta score, expected, 0.1,
          message: "Expected ~#{expected} for: #{premise} / #{hypothesis}, got: #{score}"
      end
    end

    test "all scores are between 0 and 1", %{model: model} do
      for {premise, hypothesis, _} <- @test_pairs do
        {:ok, score} = Humble.score(model, premise, hypothesis)
        assert score >= 0.0 and score <= 1.0
      end
    end
  end

  describe "score_batch/2" do
    test "batch matches individual scoring", %{model: model} do
      pairs = Enum.map(@test_pairs, fn {p, h, _} -> {p, h} end)
      {:ok, batch_scores} = Humble.score_batch(model, pairs)

      individual_scores =
        Enum.map(pairs, fn {p, h} ->
          {:ok, score} = Humble.score(model, p, h)
          score
        end)

      for {batch, individual} <- Enum.zip(batch_scores, individual_scores) do
        assert_in_delta batch, individual, 0.01
      end
    end
  end

  describe "evaluate/3" do
    test "labels consistent pair correctly", %{model: model} do
      {:ok, label} = Humble.evaluate(model, "I am in California", "I am in United States.")
      assert label == :consistent
    end

    test "labels hallucinated pair correctly", %{model: model} do
      {:ok, label} =
        Humble.evaluate(
          model,
          "The capital of France is Berlin.",
          "The capital of France is Paris."
        )

      assert label == :hallucinated
    end
  end
end
