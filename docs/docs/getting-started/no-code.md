---
sidebar_position: 2
sidebar_label: No Code
---

import 'bootstrap/dist/css/bootstrap.css';

import { useState } from 'react'
import Card from 'react-bootstrap/Card';
import Button from 'react-bootstrap/Button';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Modal from 'react-bootstrap/Modal';

import ReactTermynal from '../../src/components/termynal';

# No Code

In its simplest flavor, classy lets **you focus entirely on your data** and automatically handles the development
of full-fledged models for you, with no ML knowledge or additional line of code required. To achieve this, classy only 
asks you to specify the task you wish to tackle and to organize your data into suitable formats.

## Tasks

export const ButtonWithBackdrop = ({children, title}) => {
    const [show, setShow] = useState(false)
    const handleClose = () => setShow(false)
    const handleShow = () => setShow(true)
    return (
        <div className="mt-auto">
            <div style={{textAlign: "center"}}>
                <Button className="mt-auto" onClick={handleShow}>Show Example</Button>
            </div>
            <Modal show={show} onHide={handleClose}>
                <Modal.Header>
                    <Modal.Title>{title}</Modal.Title>
                </Modal.Header>
                <Modal.Body>{children}</Modal.Body>
            </Modal>
        </div>
    )
}

There are three tasks currently supported by classy:
<Row>
    <Col sm={4}>
        <Card className="h-100">
            <Card.Header>Sequence Classification</Card.Header>
            <Card.Body className="d-flex flex-column">
                <Card.Text style={{textAlign: "center"}}>Given a text in input (e.g. a sentence, a document), determine its most suitable label from a predefined set</Card.Text>
                <ButtonWithBackdrop title="Example: Sentiment Analysis">
                    <Row style={{alignItems: "center"}}>
                        <Col sm={7} style={{textAlign: "center"}}>I love these headphones!</Col>
                        <Col sm={1} style={{textAlign: "center"}}> &#10132; </Col>
                        <Col sm={4} style={{textAlign: "center"}}> Positive </Col>
                    </Row>
                </ButtonWithBackdrop>
            </Card.Body>
        </Card>
    </Col>
    <Col sm={4}>
        <Card className="h-100">
            <Card.Header>Sentence-Pair Classification</Card.Header>
            <Card.Body className="d-flex flex-column">
                <Card.Text style={{textAlign: "center"}}>Given two texts in input (e.g. two sentences), determine the most suitable label for this pair (usually denoting some semantic relations) from a predefined set</Card.Text>
                <ButtonWithBackdrop title="Example: Paraphrasis Detection">
                    <Row style={{alignItems: "center"}}>
                        <Col sm={7}>
                            <Row><div style={{textAlign: "center"}}><b>1.</b> I love these headphones!</div></Row>
                            <Row></Row>
                            <Row></Row>
                            <Row></Row>
                            <Row><div style={{textAlign: "center"}}><b>2.</b> I really like these headphones!</div></Row>
                        </Col>
                        <Col sm={1} style={{textAlign: "center"}}> &#10132; </Col>
                        <Col sm={4} style={{textAlign: "center"}}> Equivalent </Col>
                    </Row>
                </ButtonWithBackdrop>
            </Card.Body>
        </Card>
    </Col>
    <Col sm={4}>
        <Card className="h-100">
            <Card.Header>Token Classification</Card.Header>
            <Card.Body className="d-flex flex-column">
                <Card.Text style={{textAlign: "center"}}>Given a list of tokens, for each of them, determine its most suitable label from a predefined set</Card.Text>
                <ButtonWithBackdrop title="Example: Part-of-Speech Tagging">
                    <Row style={{alignItems: "center"}}>
                        <Col sm={3}>
                            <Row><div style={{textAlign: "center"}}>I</div></Row>
                            <Row></Row>
                            <Row><div style={{textAlign: "center"}}>&#129147;</div></Row>
                            <Row></Row>
                            <Row><div style={{textAlign: "center"}}>PRON</div></Row>
                        </Col>
                        <Col sm={3}>
                            <Row><div style={{textAlign: "center"}}>love</div></Row>
                            <Row></Row>
                            <Row><div style={{textAlign: "center"}}>&#129147;</div></Row>
                            <Row></Row>
                            <Row><div style={{textAlign: "center"}}>VERB</div></Row>
                        </Col>
                        <Col sm={3}>
                            <Row><div style={{textAlign: "center"}}>these</div></Row>
                            <Row></Row>
                            <Row><div style={{textAlign: "center"}}>&#129147;</div></Row>
                            <Row></Row>
                            <Row><div style={{textAlign: "center"}}>DET</div></Row>
                        </Col>
                        <Col sm={3}>
                            <Row><div style={{textAlign: "center"}}>headphones</div></Row>
                            <Row></Row>
                            <Row><div style={{textAlign: "center"}}>&#129147;</div></Row>
                            <Row></Row>
                            <Row><div style={{textAlign: "center"}}>NOUN</div></Row>
                        </Col>
                    </Row>
                </ButtonWithBackdrop>
            </Card.Body>
        </Card>
    </Col>
</Row>

<p></p>

:::info

These three tasks cover the vast majority of textual classification problems.

:::

:::tip

We are currently developing *genie*, classy's sibling for generation problems such as Machine Translation or Summarization. 
Contact us if you wish to partecipate to alpha testing.

:::

## Input Formats

Once you have realized which of these tasks you are dealing with, you have to organize your data. Currently, for each of 
these tasks, classy supports two possible input formats:
* .tsv
* .jsonl

:::caution

classy automatically infers how to process a given file from its extension. So please explicitly use ".tsv" for .tsv files
and ".jsonl" for .jsonl files.

:::

### .tsv

With the .tsv format, files are expected to be standard .tsv files. Their structure varies depending on the selected task:

```bash
$ cat seq.tsv | head -1
I love these headphones!    <tab>   positive     
$ cat sent-p.tsv | head -1
I love these headphones!    <tab>   I love these headphones!    <tab>   equivalent  
$ cat tok.tsv | head -1
I love these headphones !   <tab>   PRON VERB DET NOUN PUNCT
```

* Sequence Classification: two columns per line, input text and corresponding label
* Sentence-Pair Classification: three columns per line, first input text, second input text and label associated with this pair
* Token Classification: two columns per line, **space-separated** list of input tokens and **space-separated** list of corresponding labels

### .jsonl

With the .jsonl format, each line contains a JSON object that represents a sample, whose type depends on the task under 
consideration:

```bash
$ cat seq.jsonl | head -1
{"sequence": "I love these headphones!", "label": "positive"}     
$ cat sent-p.jsonl | head -1
{"sentence1": "I love these headphones!", "sentence2": "I love these headphones!", "label": "equivalent"}  
$ cat tok.jsonl | head -1
{"tokens": ["I", "love", "these", "headphones", "!"], "labels": ["PRON", "VERB", "DET", "NOUN", "PUNCT"]}
```

* Sequence Classification: object with two string fields, *sequence* and *label*
* Sentence-Pair Classification: object with three string fields, *sentence1*, *sentence2* and *label*
* Token Classification: object with two fields (each a list of strings), *tokens* and *labels*

## CLI

Once task identification and data organization are complete, your "have-to-code" section is over, and you can use classy
CLI to handle the rest, covering every step during the lifecycle of a ML model:
* train
* predict (both file-based and bash-interactive)
* evaluate
* serve via a REST API
* present a demo

### Train

The core syntax of the train command is the following:
```bash
classy train 
    <task>                # any in *{sequence,token,sentence-pair}*
    <path-to-dataset>     # path to your dataset
    -n <exp-name>         # name you want to give to your model
    -d <device>           # device on which to train ("cpu" for cpu, or device number for gpu)
```

:::info

If a device is not provided explicitly, classy will search and use a gpu if present, resorting otherwise to cpu.

:::

Your model and experiment data will be saved in *experiments/*&lt;exp-name&gt;*/current-day/current-time/*.

:::tip

If you want to transfer your model to a different pc, the simplest way is to transfer the entire 
*experiments/*&lt;exp-name&gt;*/current-day/current-time/* folder. However, multiple checkpoints, i.e. the model at different
moments in the training, might be present in the *checkpoints/* folder; to speed up the transfer, you might want to
consider moving only one of them, for instance the best one, *checkpoints/best.ckpt*.

:::

Note that *&lt;path-to-dataset&gt;* is a bit of a special parameter and can be either:
* **folder**, or, actually, a *ML-ready* folder: that is, it must contain a training file, named *train.#* 
  (# denotes a classy supported extension), and, optionally, a validation file and a test file, *validation.#* and *test.#*
* **file**: classy uses the file provided to automatically generate an ML-ready folder for you, storing it in the
  *data/* folder inside the current experiment
  
:::info

In the folder case, if *validation.#* is not present, classy automatically generates it by reserving some samples
from the training file: the new training and validation files are saved in the *data/* folder inside the current experiment.
This does not hold for *test.#* and, if not present, classy **will not** create it.

:::

<ReactTermynal>
  <span data-ty="input">classy train sequence data/output.tsv -n sequence-example</span>
  <span data-ty="progress"></span>
  <span data-ty>Training completed</span>
  <span data-ty="input">ls experiments</span>
  <span data-ty>sequence-sample/</span>
  <span data-ty="input">tree -L 2 sequence-example</span>
  <span className="data-ty-treefix" data-ty>
    <div>
        {`experiments/sequence-example/<day>/<time>/
          ├── checkpoints
            │   ├── best.ckpt
            │   ├── epoch=00-val_loss=0.51.ckpt
            │   └── last.ckpt
            ├── data
            │   ├── train.tsv
            │   └── validation.tsv
            └── ...
        `.split('\n').map( (it, i) => <p key={i} style={{lineHeight: "1.0"}}>{it}</p>)}
    </div>
  </span>
</ReactTermynal>

<p />

For all CLI commands that involve using a trained model, you can use 3 ways to specify it:
```bash
# the <exp-name> you used at training time (classy will find and use the latest best.ckpt)
classy <cmd> sequence-example ... 
# path to model folder
classy <cmd> experiments/sequence-example/ ... 
# path to specific model experiment <day>/<time> 
classy <cmd> experiments/sequence-example/<day>/<time> ... 
# path to checkpoint inside a model experiment folder
classy <cmd> experiments/sequence-example/<day>/<time>/checkpoints/epoch=00-val_loss=0.51.ckpt
```

:::tip

For all CLI commands, you can checkout `classy <command> -h` for more information and parameters.

:::

### Predict

You can use `classy predict` to perform predictions with a trained model, using 

Two modes are supported:
* **file-based prediction** 
* **bash-interactive**

File-based prediction allows you to automatically tag files. Such files can be in any supported format and need not contain 
label information: that is, the corresponding area, such as the second column for .tsv files in sequence classification, 
can be missing (if present, it will just be ignored).

<ReactTermynal>
  <span data-ty="input">cat target.tsv | head -1</span>
  <span data-ty>I wish I had never bought these terrible headphones!</span>
  <span data-ty="input">classy predict file sequence-example target.tsv -o target.out.tsv</span>
  <span data-ty="progress"></span>
  <span data-ty>Prediction complete</span>
  <span data-ty="input">cat target.out.tsv | head -1</span>
  <span data-ty>I wish I had never bought these terrible headphones!    negative</span>
</ReactTermynal>

<p />

On the other hand, bash-interactive predictions allows you to interactively query models via bash. 
Here, the input is expected to be a tab-separated string that is interpreted as the input in .tsv files.

<ReactTermynal>
  <span data-ty="input">classy predict interactive sequence-sample</span>
  <span data-ty="input" data-ty-prompt="Enter source text: ">I wish I had never bought these terrible headphones!</span>
  <span data-ty data-ty-start-delay="2000">  # prediction: negative</span>
  <span data-ty data-ty-prompt="Enter source text: "></span>
</ReactTermynal>

<p />

### Evaluate

You can use `classy evaluate` to evaluate your trained model against a dataset. If no dataset is explicitly
provided, classy will try to locate the test set provided to `classy train` (if any).

<ReactTermynal>
  <span data-ty="input">classy evaluate sequence-bert</span>
  <span data-ty="progress"></span>
  <span data-ty># accuracy: 0.9</span>
  <span data-ty># classification metrics:</span>
  <span data-ty>...</span>
  <span data-ty>    # f1: 0.9</span>
  <span data-ty>...</span>
</ReactTermynal>

<p />

:::caution

If you move your model to a different pc, automatically inferring the location of the test set will fail 
unless it was also moved (and placed in a symmetric location in the file-system). Should it fail, provide
explicitly its path in this case.

:::

### Serve

You can use `classy serve` to expose your model via a REST API with [FastAPI](https://fastapi.tiangolo.com/).

<ReactTermynal>
  <span data-ty="input">classy serve sequence-example</span>
  <span data-ty data-ty-start-delay="2000">REST API up and running at http://0.0.0.0:8000</span>
  <span data-ty>Checkout the OpenAPI docs at http://0.0.0.0:8000/docs</span>
  <span data-ty="input">curl -X 'POST' 'http://localhost:8000/' -H 'accept: application/json' -H 'Content-Type: application/json' -d '[{'{'}"sequence": "I wish I had never bought these terrible headphones!"{'}'}]'</span>
  <span data-ty data-ty-start-delay="2000">[{'{'}"sequence":"I wish I had never bought these terrible headphones!","label":"0"{'}'}]</span>
</ReactTermynal>

<p />

You can also checkout the OpenAPI documentation page we automatically generate at *http://0.0.0.0:8000/docs*.

![Classy Serve Docs](/img/intro/serve-docs.png)

:::tip

By default, `classy serve` uses port 8000. Use the *-p* parameter to specify a different one.

:::

### Present

You can use `classy demo` to spawn a [Streamlit](https://streamlit.io/) demo of your model.

<ReactTermynal>
  <span data-ty="input">classy demo sequence-example</span>
  <span data-ty data-ty-start-delay="2000">Demo up and running at http://0.0.0.0:8000</span>
</ReactTermynal>

<p />

Now you can check out out the demo at *http://0.0.0.0:8000*!

![Classy Demo](/img/intro/demo.png)

## Profiles

