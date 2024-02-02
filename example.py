import torch
import os
import datasets
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from sigma_moe import SigmaMoEForCausalLM, SigmaMoEConfiguration
from analog_moe import AnalogSigmaMoELayer, load_analog_model, save_analog_model
from aihwkit.nn.modules.linear import AnalogLinear
from aihwkit.nn.conversion import convert_to_analog
from sigma_moe.modeling_sigma_moe import SigmaMoELayer

IS_CUDA = torch.cuda.is_available()


def compute_perplexity(model: SigmaMoEForCausalLM, dataloader: DataLoader):
    """
    Compute batched perplexity.
    """
    total_loss = 0
    total_non_empty = 0
    for inputs in tqdm(dataloader):
        input_ids = inputs["input_ids"]
        input_ids = input_ids.to("cuda" if IS_CUDA else "cpu")
        labels = inputs["labels"]
        labels = labels[:, 1:].contiguous()
        labels = labels.to("cuda" if IS_CUDA else "cpu")
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            non_empty_indices = ~(labels == -100).all(1)
            logits = outputs.logits[..., :-1, :].contiguous()
            loss = torch.nn.functional.cross_entropy(
                input=logits[non_empty_indices].transpose(1, 2),
                target=labels[non_empty_indices],
                reduction="none",
            ).sum(1) / (~(labels[non_empty_indices] == -100)).sum(1)
            total_loss += loss.sum()
            total_non_empty += non_empty_indices.sum()
    mean_loss = total_loss / total_non_empty
    return mean_loss.exp()


if __name__ == "__main__":
    # first, run `huggingface-cli login` and supply a token that has the correct access rights.

    # # We load a pre-trained model and convert it to analog
    # output_dir = "/dccstor/broccoli/huggingface/transformers/sigma_moe/wikitext/moe/hw_learned_ir/"
    # model_sd = torch.load(os.path.join(output_dir, "pytorch_model.bin"))
    # for key in model_sd:
    #     if "analog_tile_state" in key:
    #         rpu_config = model_sd[key]["rpu_config"]

    # model = SigmaMoEForCausalLM.from_pretrained("ibm-aimc/sigma-moe-small")
    # model = convert_to_analog(
    #     model,
    #     rpu_config=rpu_config,
    #     conversion_map={
    #         torch.nn.Linear: AnalogLinear,
    #         SigmaMoELayer: AnalogSigmaMoELayer,
    #     },
    # )

    # # We push this model to the hub
    # save_analog_model(model, name="ibm-aimc/analog-sigma-moe-small")

    # we load it from the hub
    model = load_analog_model(
        name="ibm-aimc/analog-sigma-moe-small",
        fp_model_cls=SigmaMoEForCausalLM,
        config_cls=SigmaMoEConfiguration,
        conversion_map={
            torch.nn.Linear: AnalogLinear,
            SigmaMoELayer: AnalogSigmaMoELayer,
        },
    )

    # load the dataset. Important: This is already tokenized!
    dataset = datasets.load_dataset("ibm-aimc/sentencepiece-wikitext-103")

    # load the tokenizer (sentencepiece)
    tokenizer = AutoTokenizer.from_pretrained("ibm-aimc/sigma-moe-small")

    # create data loader from it
    dataloader = DataLoader(
        dataset=dataset["test"]["input_ids"],
        batch_size=64,
        shuffle=False,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # compute perplexity
    ppl = compute_perplexity(model, dataloader)
    print(f"Perplexity is {ppl:.2f}")
