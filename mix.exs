defmodule Humble.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/georgeguimaraes/humble"

  def project do
    [
      app: :humble,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "Humble",
      description: "HHEM hallucination detection for Elixir",
      package: package(),
      source_url: @source_url
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:bumblebee, "~> 0.6"},
      {:nx, "~> 0.9"},
      {:axon, "~> 0.7"},
      {:safetensors, "~> 0.1"},
      {:exla, "~> 0.9", optional: true},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false}
    ]
  end

  defp package do
    [
      maintainers: ["George Guimaraes"],
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url}
    ]
  end
end
