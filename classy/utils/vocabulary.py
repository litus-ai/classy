import collections
from pathlib import Path
from typing import Dict, Iterable

FIELDS_VOCABULARY_PATH = "fields_vocabulary_path.tsv"
LABELS_VOCABULARY_PATH = "labels_vocabulary_path.tsv"


class Vocabulary:

    PAD = "<pad>"
    UNK = "<unk>"

    @classmethod
    def from_samples(cls, samples: Iterable[Dict[str, str]]):
        backend_vocab = collections.defaultdict(
            lambda: {Vocabulary.PAD: 0, Vocabulary.UNK: 1}
        )
        for sample in samples:
            for k, v in sample.items():
                elem2idx = backend_vocab[k]
                if v not in elem2idx:
                    elem2idx[v] = len(elem2idx)
        return cls(backend_vocab)

    @classmethod
    def from_folder(cls, path: str):
        backend_vocab = {}
        folder = Path(path)
        for f in folder.iterdir():
            k = f.name[: f.name.rindex(".txt")]
            elem2idx = {}
            with open(f) as _f:
                for line in _f:
                    _k, _v = line.strip().split("\t")
                    elem2idx[_k] = int(_v)
            backend_vocab[k] = elem2idx
        return cls(backend_vocab)

    def __init__(self, backend_vocab: Dict[str, Dict[str, int]]):
        self.backend_vocab = backend_vocab
        self.reverse_backend_vocab = {
            k: {_v: _k for _k, _v in v.items()} for k, v in backend_vocab.items()
        }

    def get_size(self, k: str) -> int:
        return len(self.backend_vocab[k])

    def get_idx(self, k: str, elem: str) -> int:
        return self.backend_vocab[k].get(elem, self.backend_vocab[k][Vocabulary.UNK])

    def get_elem(self, k: str, idx: int) -> str:
        return self.reverse_backend_vocab[k][idx]

    def save(self, path: str) -> None:
        folder = Path(path)
        folder.mkdir()
        for k, v in self.backend_vocab.items():
            with open(folder / f"{k}.txt", "w") as f:
                for _k, _v in v.items():
                    f.write(f"{_k}\t{_v}\n")
