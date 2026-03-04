defmodule Hallmark.Model do
  @moduledoc false

  alias Bumblebee.Conversion.PyTorchParams
  alias Bumblebee.HuggingFace.Hub
  alias Bumblebee.HuggingFace.Transformers
  alias Bumblebee.Text.T5

  @hhem_repo "vectara/hallucination_evaluation_model"
  @base_repo "google/flan-t5-base"

  defstruct [
    :model,
    :params,
    :tokenizer,
    :classifier_weight,
    :classifier_bias,
    :compiler,
    max_length: 2048
  ]

  @type t :: %__MODULE__{
          model: Axon.t(),
          params: Axon.ModelState.t(),
          tokenizer: term(),
          classifier_weight: Nx.Tensor.t(),
          classifier_bias: Nx.Tensor.t(),
          compiler: module() | nil,
          max_length: pos_integer()
        }

  @doc false
  def load(opts \\ []) do
    compiler = Keyword.get(opts, :compiler)
    max_length = Keyword.get(opts, :max_length, 2048)

    with {:ok, spec} <- Bumblebee.load_spec({:hf, @base_repo}, architecture: :encoder),
         model <- Bumblebee.build_model(spec),
         {:ok, tokenizer} <- Bumblebee.load_tokenizer({:hf, @base_repo}),
         {:ok, safetensors_path} <- download_weights(),
         {:ok, params} <- load_encoder_params(spec, model, safetensors_path),
         {:ok, {weight, bias}} <- load_classifier_weights(safetensors_path) do
      {:ok,
       %__MODULE__{
         model: model,
         params: params,
         tokenizer: tokenizer,
         classifier_weight: weight,
         classifier_bias: bias,
         compiler: compiler,
         max_length: max_length
       }}
    end
  end

  @doc false
  def predict(%__MODULE__{} = model, premise, hypothesis) do
    case predict_batch(model, [{premise, hypothesis}]) do
      {:ok, [score]} -> {:ok, score}
      error -> error
    end
  end

  @doc false
  def predict_batch(%__MODULE__{} = model_data, pairs) do
    %{
      model: model,
      params: params,
      tokenizer: tokenizer,
      classifier_weight: weight,
      classifier_bias: bias,
      compiler: compiler
    } = model_data

    texts =
      Enum.map(pairs, fn {premise, hypothesis} ->
        format_prompt(premise, hypothesis)
      end)

    configured_tokenizer =
      Bumblebee.configure(tokenizer, length: model_data.max_length, pad_direction: :right)

    inputs = Bumblebee.apply_tokenizer(configured_tokenizer, texts)

    predict_opts = if compiler, do: [compiler: compiler], else: []
    %{hidden_state: hidden_state} = Axon.predict(model, params, inputs, predict_opts)

    # Position 0 is the <pad> token, used as CLS for classification
    cls = hidden_state[[.., 0, ..]]

    # Classifier head: logits = cls @ weight^T + bias
    logits = Nx.add(Nx.dot(cls, Nx.transpose(weight)), bias)

    # Softmax → index 1 is the "consistent" probability
    scores = softmax(logits)[[.., 1]]

    {:ok, Nx.to_flat_list(scores)}
  end

  def format_prompt(premise, hypothesis) do
    "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: #{premise}\n\nHypothesis: #{hypothesis}"
  end

  defp download_weights do
    url = Hub.file_url(@hhem_repo, "model.safetensors", nil)
    Hub.cached_download(url, cache_scope: @hhem_repo)
  end

  defp load_encoder_params(spec, model, safetensors_path) do
    params_mapping = Transformers.Model.params_mapping(spec)

    # HHEM prefixes all encoder weights with "t5.transformer."
    hhem_mapping =
      Map.new(params_mapping, fn {axon_name, pytorch_name} ->
        {axon_name, "t5.transformer.#{pytorch_name}"}
      end)

    input_template = T5.input_template(spec)

    params =
      PyTorchParams.load_params!(
        model,
        input_template,
        safetensors_path,
        params_mapping: hhem_mapping,
        loader_fun: &Safetensors.read!(&1, lazy: true)
      )

    {:ok, params}
  rescue
    e -> {:error, Exception.message(e)}
  end

  defp load_classifier_weights(safetensors_path) do
    tensors = Safetensors.read!(safetensors_path)

    weight = Map.get(tensors, "t5.classifier.weight")
    bias = Map.get(tensors, "t5.classifier.bias")

    if weight && bias do
      {:ok, {weight, bias}}
    else
      {:error, "classifier weights not found in HHEM safetensors"}
    end
  rescue
    e -> {:error, Exception.message(e)}
  end

  defp softmax(logits) do
    max = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    exp = Nx.exp(Nx.subtract(logits, max))
    Nx.divide(exp, Nx.sum(exp, axes: [-1], keep_axes: true))
  end
end
