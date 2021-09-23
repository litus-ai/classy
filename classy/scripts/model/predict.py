import argparse

import torch

from classy.data.data_drivers import get_data_driver, TSV
from classy.utils.lightning import load_classy_module_from_checkpoint, load_prediction_dataset_conf_from_checkpoint


def interactive_main(
    model_checkpoint_path: str,
    cuda_device: int,
):

    model = load_classy_module_from_checkpoint(model_checkpoint_path)
    model.to(torch.device(cuda_device if cuda_device != -1 else "cpu"))
    model.freeze()

    dataset_conf = load_prediction_dataset_conf_from_checkpoint(model_checkpoint_path)
    data_driver = get_data_driver(model.task, TSV)

    while True:
        source = input("Enter source text: ").strip()
        _, prediction = next(
            model.predict(
                data_driver.read([source]),
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
        for source, prediction in model.predict(
            data_driver.read_from_path(input_path),
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
    # file params
    parser.add_argument("-f", type=str, default=None, help="Input file")
    parser.add_argument("-o", type=str, default=None, help="Output file")
    parser.add_argument("--token-batch-size", type=int, default=128, help="Token batch size")
    # return
    return parser.parse_args()


if __name__ == "__main__":
    main()
