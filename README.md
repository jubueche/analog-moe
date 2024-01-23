# Analog MoE
## Requirements
You need to have a GPU which is at least Volta (V100, A100, H100) since this package leverages triton.
## Getting started 🚀
You can create a clean environment using the following
```
conda create -n torch-nightly python=3.10 -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
conda install -c conda-forge aihwkit-gpu -y
pip install triton
```
Now, you should be able to call `python test_moe_layer.py` and the script should exit without any errors.

## Usage ⚒️
You can convert any aihwkit model and swap out the `SigmaMoELayer`s like so:
```
model = convert_to_analog(
    model,
    rpu_config=<some_rpu_config>,
    conversion_map={
        torch.nn.Linear: AnalogLinear,
        SigmaMoELayer: AnalogSigmaMoELayer
    }
)
```
For a full example, see [here](https://github.com/jubueche/Sigma-MoE/blob/main/train.py).