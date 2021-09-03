import argparse
from typing import List, Iterable, Tuple, Generator, Dict, Union, Optional

import hydra.utils
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from classy.data.data_drivers import DataDriver, get_data_driver, TSV, SentencePairSample, SequenceSample, TokensSample
from classy.pl_modules.base import ClassyPLModule
from classy.utils.lightning import load_classy_module_from_checkpoint, load_prediction_dataset_conf_from_checkpoint


def predict(
    model: ClassyPLModule,
    sources: Iterable[str],
    data_driver: DataDriver,
    dataset_conf: Dict,
    token_batch_size: int = 1024,
    progress_bar: bool = False,
) -> Generator[Tuple[Union[SentencePairSample, SequenceSample, TokensSample], Union[str, List[str]]], None, None]:

    # todo only works on single gpu
    device = next(model.parameters()).device

    # instantiate dataset
    dataset_conf["tokens_per_batch"] = token_batch_size
    dataset = hydra.utils.instantiate(dataset_conf, lines=sources, data_driver=data_driver, vocabulary=model.vocabulary)

    # instantiate dataloader
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    iterator = dataloader
    if progress_bar:
        iterator = tqdm(iterator, desc="Predicting")

    for batch in iterator:

        # predict
        with autocast(enabled=True):
            with torch.no_grad():
                batch_out = model.predict(**{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()})

                for sample, prediction in batch_out:
                    yield sample, prediction


def interactive_main(
    model_checkpoint_path: str,
    cuda_device: int,
):

    model = load_classy_module_from_checkpoint(model_checkpoint_path)
    model.to(torch.device(cuda_device if cuda_device != -1 else "cpu"))
    model.eval()
    model.freeze()

    dataset_conf = load_prediction_dataset_conf_from_checkpoint(model_checkpoint_path)
    data_driver = get_data_driver(model.task, TSV)

    while True:
        source = input("Enter source text: ").strip()
        _, prediction = next(
            predict(
                model,
                [source],
                data_driver=data_driver,
                dataset_conf=dataset_conf,
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

    dataset_conf = load_prediction_dataset_conf_from_checkpoint(model_checkpoint_path)
    input_extension, output_extension = input_path.split(".")[-1], output_path.split(".")[-1]
    assert input_extension == output_extension, (
        f"Having different input and output extensions is not currently a supported use case: "
        f"input {input_extension} != output {output_extension}"
    )
    data_driver = get_data_driver(model.task, input_extension)

    def it():
        with open(input_path) as fi:
            for source, prediction in predict(
                model,
                map(lambda l: l.strip(), fi),
                data_driver=data_driver,
                token_batch_size=token_batch_size,
                dataset_conf=dataset_conf,
                progress_bar=True,
            ):
                source.update_classification(prediction)
                yield source

    data_driver.save(it(), output_path)


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
