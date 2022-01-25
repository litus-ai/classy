---
sidebar_position: 7
title: Custom Evaluation Metric
---

import ReactTermynal from '/src/components/termynal';

Adding a custom metric for evaluation is easy in `classy`, and you can use it for both `classy evaluate` and
`classy train` (to monitor performance or, perhaps, even early-stop). To do this, you just need to:

1. Write your *Evaluation* class

```python
class Evaluation:
    def __call__(
        self,
        predicted_samples: List[ClassySample],
    ) -> Dict:
        raise NotImplementedError
```

2. Write its config
3. Train, specifying your evaluation `classy train [...] -c [...] evaluation=<your evaluation name>`
4. using `classy evaluate` now prints your custom evaluation

## A Minimal Example

As an example, imagine you want to use SpanF1 to evaluate your NER (Named Entity Recognition) system. First, you implement
the class:
```python title="classy/evaluation/span.py"
from datasets import load_metric

class SeqEvalSpanEvaluation(Evaluation):
    def __init__(self):
        self.backend_metric = load_metric("seqeval")

    def __call__(
        self,
        predicted_samples: List[TokensSample],
    ) -> Dict:

        metric_out = self.backend_metric.compute(
            predictions=[labels for _, labels in predicted_samples],
            references=[sample.labels for sample, _ in predicted_samples],
        )
        p, r, f1 = metric_out["overall_precision"], metric_out["overall_recall"], metric_out["overall_f1"]

        return {"precision": p, "recall": r, "f1": f1}
```
We use here the SpanF1 metric implemented in the HuggingFace *datasets* library (this is what *load_metric("seqeval")*
does). Then, you write the corresponding config:
```yaml title="configurations/evaluation/span.yaml"
_target_: 'classy.evaluation.span.SeqEvalSpanEvaluation'
```

Finally, add this evaluation metric to your training configuration, train your model and automatically evaluate with
your metric:

<ReactTermynal>
  <span data-ty="input">classy train token DATA-PATH -n token -c evaluation=span</span>
  <span data-ty="progress"></span>
  <span data-ty>Training completed</span>
  <span data-ty="input">classy evaluate MODEL-PATH TEST-PATH</span>
  <span data-ty="progress"></span>
  <span data-ty>* precision: 0.8746950156849076</span>
  <span data-ty>* recall: 0.8886331444759207</span>
  <span data-ty>* f1: 0.8816089935007905</span>
</ReactTermynal>


## Monitoring at Training Time

As a matter of fact, most of the time you'll want to monitor your evaluation metric on some dataset (most likely, the validation)
also during training. You can achieve this as follows:

<ReactTermynal>
  <span data-ty="input">classy train token DATA-PATH -n token -c callbacks=evaluation evaluation=span</span>
  <span data-ty="progress"></span>
  <span data-ty>Training completed</span>
</ReactTermynal>
<p></p>

`callbacks=evaluation` is what does the magic. Behind the scenes, what is happening is that you are adding a callback
with the following config (which, obviously, you can modify either with `-c` or via profile):
```yaml title="configurations/callbacks/evaluation.yaml"
- _target_: "classy.pl_callbacks.prediction.PredictionPLCallback"
  path: null  # leave it to null to set it to validation path
  prediction_dataset_conf: ${prediction.dataset}
  on_result:
    file_dumper:
      _target_: "classy.pl_callbacks.prediction.FileDumperPredictionCallback"
    evaluation:
      _target_: "classy.pl_callbacks.prediction.EvaluationPredictionCallback"
      evaluation: ${evaluation}
  settings:
    - name: "validation"
      path: null  # leave it to null to set it to PredictionPLCallback.path
      token_batch_size: 800
      limit: 1000
      prediction_param_conf_path: null
      on_result:
        - "file_dumper"
        - "evaluation"
```
Left as it is, this config tells `classy` to use the model being trained to predict all samples in the validation dataset,
and runs 2 callbacks on the resulting (sample, prediction) tuples:
* *FileDumperPredictionCallback*; this callback dumps the (sample, prediction) tuples that your model predicts at each
validation epoch in a dedicated folder in your experiment directory
* *EvaluationPredictionCallback* (the actual magic); this callback evaluates the (sample, prediction) tuples with the
evaluation metric you specified and logs the result

More in detail, *PredictionPLCallback* is a powerful class supporting quite the number of evaluation scenarios during
your training. It has 2 main arguments:
* *on_result*, a dictionary of (name, callback) pairs; each callback here is a *classy.pl_callbacks.prediction.PredictionCallback* class
* *settings*, a list of settings where model prediction should be performed, each made up of:
  * *name*
  * *path* (containing the dataset you want to evaluate upon)
  * *token_batch_size*, the token batch size you want to use (remember, no gradient computation here)
  * *limit*, the maximum number of samples to be used (chosen as they occur in the dataset); set it to -1 to use all of them
  * *prediction_param_conf_path*, the path to the prediction params config file you want to use (leave it to null if not needed)
  * (optionally) *on_result*, a list containing the name of the on_result callbacks to want to launch on this setting; if not
provided, all callbacks will be used

:::tip

You can use your metric for early-stopping as well! Just add
`-c [...] callbacks_monitor=<setting-name>-<name-of-metric-returned-in-evaluation-dict> callbacks_mode=<max-or-min>`.
For instance, in our example, to early-stop on SpanF1 on the validation set,
use `-c [...] callbacks_monitor=validation-f1 callbacks_mode=max`

:::

## Swapping Evaluation Metric

`classy` also supports changing the evaluation metric directly when using `classy evaluate`, regardless of the config
used in `classy train`. To do so, you can use the the `--evaluation-config` CLI parameter to `classy evaluate`. This
parameter specifies the configuration path (e.g. *configurations/evaluation/span.yaml*) where the config of the desired
evaluation metric is stored.

<ReactTermynal>
  <span data-ty="input">classy train token DATA-PATH -n token</span>
  <span data-ty="progress"></span>
  <span data-ty>Training completed</span>
  <span data-ty="input">classy evaluate MODEL-PATH TEST-PATH</span>
  <span data-ty="progress"></span>
  <span data-ty># Evaluation with original training config</span>
  <span data-ty>[...]</span>
  <span data-ty="input">classy evaluate MODEL-PATH TEST-PATH --evaluation-config configurations/evaluation/span.yaml</span>
  <span data-ty="progress"></span>
  <span data-ty>* precision: 0.8746950156849076</span>
  <span data-ty>* recall: 0.8886331444759207</span>
  <span data-ty>* f1: 0.8816089935007905</span>
</ReactTermynal>
<p></p>

:::caution

Note that interpolation to other configs is currently not supported in this setting.

:::
