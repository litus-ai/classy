import argparse
from typing import List, Iterable, Tuple, Optional, Generator

import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from classy.data.dataset.hf import HFSequenceDataset
from classy.data.readers import Reader, get_reader, TSV
from classy.utils.lightning import load_classy_module_from_checkpoint

from classy.pl_modules.base import ClassyPLModule


def predict(
    model: ClassyPLModule,
    sources: Iterable[str],
    reader: Reader,
    dataset_conf: omegaconf.OmegaConf,
    token_batch_size: int = 1024,
    progress_bar: bool = False,
) -> Generator[Tuple[str, str, Optional[str]], None, None]:

    # todo only works on single gpu
    device = next(model.parameters()).device

    # instantiate dataset
    # todo remove coupling
    dataset = HFSequenceDataset.from_lines(
        sources,
        reader=reader,
        vocabulary=model.vocabulary,
        transformer_model='bert-large-cased',
        min_length=5,
        max_length=500,
        tokens_per_batch=800,
        max_batch_size=10,
        section_size=10000,
        prebatch=True,
        shuffle=True,
    )
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    iterator = dataloader
    if progress_bar:
        iterator = tqdm(iterator, desc="Predicting")

    for batch in iterator:

        # predict
        with autocast(enabled=True):
            with torch.no_grad():
                batch_predictions = model.predict(
                    **{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                )

                # todo here we should yield (<input>, predicted label, gold label)
                for p in batch_predictions:
                    yield '', p, None


def interactive_main(
    model_checkpoint_path: str,
    cuda_device: int,
):

    model = load_classy_module_from_checkpoint(model_checkpoint_path)
    model.to(torch.device(cuda_device if cuda_device != -1 else "cpu"))
    model.eval()
    model.freeze()

    while True:
        source = input("Enter source text: ").strip()
        _, prediction, _ = next(
            predict(
                model,
                [source],
                reader=get_reader(model.task, TSV),
                dataset_conf=None
            )
        )
        print(f"\t# prediction: \t{prediction}")


def file_main(
    model_checkpoint_path: str,
    input_path: str,
    output_path: str,
    cuda_device: int,
    token_batch_size: int,
):

    model = load_classy_module_from_checkpoint(model_checkpoint_path)
    model.to(torch.device(cuda_device if cuda_device != -1 else "cpu"))
    model.eval()
    model.freeze()

    with open(input_path) as fi, open(output_path, "w") as fo:
        for source, prediction, _ in predict(
            model,
            map(lambda l: l.strip(), fi),
            reader=get_reader(model.task, input_path.split(".")[-1]),
            token_batch_size=token_batch_size,
            dataset_conf=None,
            progress_bar=True,
        ):
            # todo we are dumping tsv even if the output was a jsonl, convert readers into io-handlers?
            fo.write(f"{source}\t{prediction.strip()}\n")


def main():
    args = parse_args()
    if args.t:
        interactive_main(
            args.model_checkpoint,
            cuda_device=args.cuda_device,
        )
    else:
        file_main(
            args.model_checkpoint,
            args.f,
            args.o,
            cuda_device=args.cuda_device,
            token_batch_size=args.token_batch_size,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_checkpoint", type=str, help="Path to pl_modules checkpoint")
    parser.add_argument("--cuda-device", type=int, default=-1, help="Cuda device")
    # interactive params
    parser.add_argument("-t", action="store_true", help="Interactive mode")
    # generation params
    parser.add_argument("-f", type=str, default=None, help="Input file")
    parser.add_argument("-o", type=str, default=None, help="Output file")
    parser.add_argument("--token-batch-size", type=int, default=128, help="Token batch size")
    # return
    return parser.parse_args()


if __name__ == "__main__":
    main()
