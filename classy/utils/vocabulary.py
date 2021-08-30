from typing import Iterable, Dict


FIELDS_VOCABULARY_PATH = "fields_vocabulary_path.tsv"
LABELS_VOCABULARY_PATH = "labels_vocabulary_path.tsv"


class Vocabulary:

    PAD = "<pad>"
    UNK = "<unk>"

    @classmethod
    def from_samples(cls, samples: Iterable[str]):
        elem2idx = {Vocabulary.PAD: 0, Vocabulary.UNK: 1}
        for sample in samples:
            if sample not in elem2idx:
                elem2idx[sample] = len(elem2idx)
        return cls(elem2idx)

    @classmethod
    def from_file(cls, file_path: str):
        elem2idx = {}
        with open(file_path) as f:
            for line in f:
                key, value = line.strip().split()
                value = int(value)
                elem2idx[key] = value
        return cls(elem2idx)

    def __init__(self, elem2idx: Dict[str, int]):
        self.elem2idx = elem2idx
        self.idx2elem = {v: k for k, v in elem2idx.items()}

    def get_idx(self, elem: str) -> int:
        return self.elem2idx.get(elem, self.elem2idx[Vocabulary.UNK])

    def get_elem(self, idx: int) -> str:
        return self.idx2elem[idx]

    def save(self, file_path: str) -> None:
        with open(file_path, "w") as f:
            for key, value in self.elem2idx.items():
                f.write(f"{key}\t{value}\n")
