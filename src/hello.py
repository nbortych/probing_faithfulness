from models.transformers_model import TransformersModel
# Initialize model with activation collection
model = TransformersModel(
    model_name="ComCom/gpt2-small",
    activation_points=[
        "transformer.h.0",  # First transformer layer
        "transformer.h.11.mlp",  # Last layer MLP
    ]
)

# Use as context manager for automatic cleanup
with model as m:
    # Get outputs and activations
    results = m.forward("Testing model faithfulness")
    print(results.keys())
    # Access activations
    first_layer_activations = results['activations']['transformer.h.0']
    last_mlp_activations = results['activations']['transformer.h.11.mlp']
    print(first_layer_activations[0].shape)
    print(last_mlp_activations[0].shape)