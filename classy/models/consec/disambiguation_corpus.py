import collections
import logging
import re
import string

from typing import NamedTuple, Optional, List, Iterator, Tuple, Generator, Callable, Any

from classy.data.data_drivers import TokensSample
from classy.models.consec.utils import pos_map

logger = logging.getLogger(__name__)


class DisambiguationInstance(NamedTuple):
    document_id: str
    sentence_id: str
    instance_id: Optional[str]
    text: str
    pos: str
    lemma: str
    labels: Optional[List[str]]
    token_sample: TokensSample


def access_sparse_zipped_index(tgt: int, keys: List[int], values: List[Any]) -> Optional[Any]:
    try:
        return values[keys.index(tgt)]
    except:
        pass
    return None


class DisambiguationCorpus:
    @classmethod
    def from_samples(cls, samples: Iterator[TokensSample], **kwargs):
        def r():
            for sample in samples:
                disambiguation_sentences = []
                for i in range(len(sample.tokens)):
                    di_labels = (
                        access_sparse_zipped_index(i, sample.target, sample.labels)
                        if sample.labels is not None
                        else None
                    )
                    disambiguation_sentences.append(
                        DisambiguationInstance(
                            document_id=sample.document_id,
                            sentence_id=sample.sentence_id,
                            instance_id=access_sparse_zipped_index(i, sample.target, sample.instance_ids),
                            text=sample.tokens[i],
                            pos=pos_map.get(sample.pos[i], sample.pos[i]),
                            lemma=sample.lemma[i],
                            labels=di_labels.split(" --- ") if di_labels is not None else None,
                            token_sample=sample,
                        )
                    )
                yield disambiguation_sentences

        return cls(r, **kwargs)

    def __init__(self, corpus_iterator: Callable[[], Iterator[List[DisambiguationInstance]]], is_doc_based: bool):
        self.corpus_iterator = corpus_iterator
        self.is_doc_based = is_doc_based
        if is_doc_based:
            logger.warning("Corpus was flagged as doc_based => this requires corpus materialization")
            self.sentences = []
            self.doc2sent_pos = collections.defaultdict(dict)
            self.doc2sent_order = collections.defaultdict(list)
            self.sentences_index = dict()
            self._load_corpus_indexing_structures()

    def _load_corpus_indexing_structures(self) -> None:
        logger.info("Initializing corpus indexing structures")

        self._disambiguation_sentences = []

        for disambiguation_sentence in self.corpus_iterator():

            self._disambiguation_sentences.append(disambiguation_sentence)

            sentence_rep = disambiguation_sentence[0]
            sentence_doc_id = sentence_rep.document_id
            sentence_id = sentence_rep.sentence_id

            self.doc2sent_order[sentence_doc_id].append(sentence_id)
            self.sentences_index[sentence_id] = disambiguation_sentence

        for document_id in self.doc2sent_order.keys():
            ordered_sentences = sorted(self.doc2sent_order[document_id])
            for sentence_index, sentence_id in enumerate(ordered_sentences):
                self.doc2sent_pos[document_id][sentence_id] = sentence_index
            self.doc2sent_order[document_id] = ordered_sentences

        self.corpus_iterator = lambda: self._disambiguation_sentences
        logger.info("Corpus indexing structures completed")

    def __iter__(self) -> Generator[List[DisambiguationInstance], None, None]:
        for sentence in self.corpus_iterator():
            if re.fullmatch(rf"[{string.punctuation}]*", sentence[-1].text) is None:
                sentence.append(
                    DisambiguationInstance(
                        sentence[0].document_id,
                        sentence[0].sentence_id,
                        None,
                        ".",
                        pos_map.get("PUNCT", "PUNCT"),
                        ".",
                        None,
                        token_sample=sentence[-1].token_sample,
                    )
                )
                logger.debug(
                    f'Found sentence with missing trailing punctuation, adding it: {" ".join([di.text for di in sentence])}'
                )
            yield sentence

    def get_neighbours_sentences(
        self, document_id: str, sentence_id: str, prev_sent_num: int, next_sent_num: int
    ) -> Tuple[List[List[DisambiguationInstance]], List[List[DisambiguationInstance]]]:

        if not self.is_doc_based:
            return [], []

        sentence_position_in_doc = self.doc2sent_pos[document_id][sentence_id]
        doc_sentences_id = self.doc2sent_order[document_id]

        prev_sentences_id = doc_sentences_id[
            max(sentence_position_in_doc - prev_sent_num, 0) : sentence_position_in_doc
        ]
        next_sentences_id = doc_sentences_id[
            sentence_position_in_doc + 1 : sentence_position_in_doc + 1 + next_sent_num
        ]

        prev_sentences = [self.sentences_index[sent_id] for sent_id in prev_sentences_id]
        next_sentences = [self.sentences_index[sent_id] for sent_id in next_sentences_id]

        return prev_sentences, next_sentences

    def __len__(self):
        return len(self.sentences)
