

from act_models import ModelDETRVAE

config = {
    # Transformer
    "hidden_dim": 256,
    "dropout": 0.1,
    "nheads": 8,
    "dim_feedforward": 2048,
    "enc_layers": 6,
    "dec_layers": 6,
    "pre_norm": False,
    "return_intermediate_dec": True,

    # Backbone
    "backbone": "resnet50",
    "lr_backbone": 0.1,
    "masks": True,
    "dilation": False,

    # Position Encoding
    "position_embedding": "sine",
    
    # other parameters
    "state_dim": 14,
    "num_queries": 100,
    "camera_names": ['cam1', 'cam2'],
    "use_image": True
}

model_builder = ModelDETRVAE()

model = model_builder.build(config)
print(model)



