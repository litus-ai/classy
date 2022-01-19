---
sidebar_position: 4
title: Custom Model
---


:::info

`classy` is built on top of PyTorch Lightning and, in order to better understand classy code infrastructure, we recommend
going through PyTorch Lightning [intro guide](https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html)
before proceeding.

:::

Implementing your own model within classy is easy. You just need to:
* subclass `ClassyPLModule` and your task mixin (*SequenceTask*, *SentencePairTask*, *TokensTask*, *QATask*)
* implement abstract methods
* (optional) override any other method

For instance, considering Sequence Classification, you would need to implement the following class:

```python
# subclass your task and ClassyPLModule
class MyCustomClassyPLModule(SequenceTask, ClassyPLModule):
    def __init__(
        self,
        param1: Any,
        param2: Any,
        vocabulary: Vocabulary,
        optim_conf: omegaconf.DictConfig,
    ):
        super().__init__(vocabulary=vocabulary, optim_conf=optim_conf)
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        # standard pytorch forward
        raise NotImplementedError

    def batch_predict(self, *args, **kwargs) -> Iterator[Tuple[SequenceSample, str]]:
        # wrapper for your forward method
        # it takes as input the batches produced by your dataset
        # it emits tuples (sequence sample, predicted label)
        # decoding logic, such as converting labels from tensors to strings, goes here
        raise NotImplementedError

    ###################
    # lightning hooks #
    ###################

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        raise NotImplementedError

    def test_step(self, batch: dict, batch_idx: int) -> None:
        raise NotImplementedError
```

## A Minimal Example

Practically, imagine you want to build a Sequence Classification model on top of a HuggingFace Transformer model.

```python title="classy/pl_modules/custom_model.py"
class MyCustomClassyPLModule(SequenceTask, ClassyPLModule):
    pass
```

You first implement its constructor:
```python
def __init__(
    self,
    transformer_model: str,
    vocabulary: Vocabulary,
    optim_conf: omegaconf.DictConfig,
):
    super().__init__(vocabulary=vocabulary, optim_conf=optim_conf)
    self.save_hyperparameters(ignore="vocabulary")
    num_classes = vocabulary.get_size(k="labels")  # number of target classes
    self.classifier = AutoModelForSequenceClassification.from_pretrained(
        transformer_model, num_labels=num_classes
    )  # underlying classifier
    self.accuracy_metric = (
        torchmetrics.Accuracy()
    )  # metric to track your model performance
```

Then, you need to implement the PyTorch forward:

```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    samples: List[SequenceSample],
    token_type_ids: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> ClassificationOutput:
    model_input = {"input_ids": input_ids, "attention_mask": attention_mask}
    if token_type_ids is not None:
        model_input["token_type_ids"] = token_type_ids
    if labels is not None:
        model_input["labels"] = labels
    model_output = self.classifier(**model_input)
    return ClassificationOutput(
        logits=model_output.logits,
        probabilities=torch.softmax(model_output.logits, dim=-1),
        predictions=torch.argmax(model_output.logits, dim=-1),
        loss=model_output.loss,
    )
```

There's nothing really special about this forward. `ClassificationOutput` is just a dataclass to conveniently store logits,
probabilities, predictions and loss. The only important thing is the signature: it **must match** with the batches your
dataset emits (here, we are using *classy.data.dataset.hf.HFSequenceDataset*).

Then, there's the batch predict method, which wraps your forward method to emit classified *SequenceSample*-s:

```python
def batch_predict(
    self, *args, **kwargs
) -> Iterator[Tuple[Union[SequenceSample, SentencePairSample], str]]:
    samples = kwargs.get("samples")
    classification_output = self.forward(*args, **kwargs)
    for sample, prediction in zip(samples, classification_output.predictions):
        yield sample, self.vocabulary.get_elem(k="labels", idx=prediction.item())
```

You just invoke the forward method, and use the vocabulary to perform label tensor-to-string decoding.

Finally, you have to implement lightning hooks:

```python
def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
    classification_output = self.forward(**batch)
    self.log("loss", classification_output.loss)
    return classification_output.loss


def validation_step(self, batch: dict, batch_idx: int) -> None:
    classification_output = self.forward(**batch)
    self.accuracy_metric(classification_output.predictions, batch["labels"].squeeze(-1))
    self.log("val_loss", classification_output.loss)
    self.log("val_accuracy", self.accuracy_metric, prog_bar=True)


def test_step(self, batch: dict, batch_idx: int) -> None:
    classification_output = self.forward(**batch)
    self.accuracy_metric(classification_output.predictions, batch["labels"].squeeze(-1))
    self.log("test_accuracy", self.accuracy_metric)
```

The only missing component is writing the configuration file:

```yaml title="model/sequence-custom.yaml"
_target_: 'classy.pl_modules.custom_model.MyCustomClassyPLModule'
transformer_model: ${transformer_model}
optim_conf:
  _target_: classy.optim.factories.TorchFactory
  optimizer:
    _target_: torch.optim.Adam
    lr: 1e-5
```

and start the training:

```bash
classy train sequence <dataset-path> -c model=sequence-custom
```
