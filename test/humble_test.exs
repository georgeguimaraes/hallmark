defmodule HumbleTest do
  use ExUnit.Case

  alias Humble.Model

  describe "format_prompt/2" do
    test "formats premise and hypothesis into HHEM prompt" do
      prompt = Model.format_prompt("I am in California", "I am in United States.")

      assert prompt ==
               "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: I am in California\n\nHypothesis: I am in United States."
    end

    test "handles special characters" do
      prompt = Model.format_prompt("It's 100% true.", "That's \"correct\".")

      assert prompt ==
               "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: It's 100% true.\n\nHypothesis: That's \"correct\"."
    end
  end

  describe "threshold logic" do
    test "score at threshold is consistent" do
      assert classify(0.5, 0.5) == :consistent
    end

    test "score above threshold is consistent" do
      assert classify(0.8, 0.5) == :consistent
    end

    test "score below threshold is hallucinated" do
      assert classify(0.3, 0.5) == :hallucinated
    end

    test "custom threshold" do
      assert classify(0.7, 0.8) == :hallucinated
      assert classify(0.9, 0.8) == :consistent
    end

    defp classify(score, threshold) do
      if score >= threshold, do: :consistent, else: :hallucinated
    end
  end
end
