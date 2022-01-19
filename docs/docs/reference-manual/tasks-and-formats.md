---
sidebar_position: 2
title: Tasks and Input Formats
---

import ReactTermynal from '/src/components/termynal';

In its simplest flavor, `classy` lets **you focus entirely on your data** and automatically handles the development
of full-fledged models for you, with no ML knowledge or additional line of code required. To achieve this, `classy` only
asks you to specify the task you wish to tackle and to organize your data into suitable formats.

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

There are five tasks currently supported by classy:
* Sequence Classification
* Sentence-Pair Classification
* Token Classification
* Text Extraction
* Generation

:::info

These tasks cover the vast majority of NLP problems.

:::

#### Data Formats

Once you have realized which task you are dealing with, you have to organize your data. Currently, for each of
these tasks, `classy` supports two possible input formats:
* `.tsv`: files are expected to be standard TSV (tab-separated values) files.
* `.jsonl`: files are expected to contain, **for each line**, a JSON object representing a sample.

:::caution

`classy` automatically infers how to process a given file from its extension. So please explicitly use `".tsv"` for `.tsv` files
and `".jsonl"` for `jsonl` files.

:::

:::info
`classy` supports the addition of [custom data formats](/docs/getting-started/customizing-things/custom-data-format/).
:::

### Sequence Classification

Given a text in input (e.g. a sentence, a document), determine its most suitable label from a predefined set.

:::tip

An example is *Sentiment Analysis*:
* **input**: I love these headphones!
* **output**: positive

:::

#### TSV

```bash
# two columns per line:
# * input text
# * corresponding label
$ cat seq.tsv | head -1
I love these headphones!    <tab>   positive
```

#### JSONL

```bash
# object with fields:
# * sequence (string)
# * label (string)
$ cat seq.jsonl | head -1
{"sequence": "I love these headphones!", "label": "positive"}
```

### Sentence-Pair Classification

Given two texts in input (e.g. two sentences), determine the most suitable label for this pair (usually denoting some semantic relations) from a predefined set.

:::tip

An example is *Paraphrasis Detection*:
* **first input sentence**: I love these headphones!
* **second input sentence**: I really like these headphones!
* **output**: equivalent

:::

#### TSV

```bash
# three columns per line:
# * first input text
# * second input text
# * label
$ cat sent-p.tsv | head -1
I love these headphones!    <tab>   I love these headphones!    <tab>   equivalent
```

#### JSONL

```bash
# object with fields:
# * sentence1 (string)
# * sentence2 (string)
# * label (string)
$ cat sent-p.jsonl | head -1
{"sentence1": "I love these headphones!", "sentence2": "I love these headphones!", "label": "equivalent"}
```

## Token Classification

Given a list of tokens, for each of them, determine its most suitable label from a predefined set.

:::tip

An example is *Part-of-Speech Tagging*:
* **input**: I love these headphones
* **output**: PRON VERB DET NOUN

:::

#### TSV

```bash
# two columns per line:
# * space-separated list of input tokens
# * space-separated list of labels
$ cat tok.tsv | head -1
I love these headphones !   <tab>   PRON VERB DET NOUN PUNCT
```

#### JSONL

```bash
# object with fields:
# * tokens (list of strings)
# * labels (list of strings)
$ cat tok.jsonl | head -1
{"tokens": ["I", "love", "these", "headphones", "!"], "labels": ["PRON", "VERB", "DET", "NOUN", "PUNCT"]}
```

## Text Extraction

Given a context (e.g. a document) and some query about this document (e.g. a question), extract the consecutive span in the context that best addresses the query (e.g. an answer).

:::tip

An example is *Question Answering*:
* **input context**: I love these headphones!
* **input query**: What do you love?
* **output**: I love <mark style={{backgroundColor: "#f1e740"}}>these headphones</mark>!

:::

#### TSV

```bash
# four columns per line:
# * context
# * query
# * start char offset
# * end char offset
$ cat text-ext.tsv | head -1
I love these headphones!    <tab>   What do you love?           <tab>   7  <tab>  23
```

#### JSONL

```bash
# object with fields:
# * context (string)
# * query (string)
# * answer_start (int) as a char offset
# * answer_end (int) as a char offset
$ cat text-ext.jsonl | head -1
{"context": "I love these headphones!", "query": "What do you love?", "answer_start", 7, "answer_end": 23}
```

## Generation

Generally speaking, generation is the task of *generating* a target text given a source text in input (e.g. a sentence).


:::tip

An example is *English-to-Italian Machine Translation*:
* **input**: I love these headphones!
* **output**: Io adoro queste cuffie!

:::

:::caution

The input format depends on whether you wish to do standard generation or language modeling:
* **standard generation**: given a source text (e.g. *I loved these headphones!*), generate the corresponding target text (e.g., in Italian, *Io adoro queste cuffie!*)
* **language modeling**: given a text (e.g. *I came back home because*), generate the subsequent words (e.g. *I had forgotten my wallet*)

:::

#### TSV

```bash
# [standard generation] two columns per line:
# * source text
# * target text
$ cat gen.tsv | head -1
I love these headphones!    <tab>   Io adoro queste cuffie!

# [language modeling] one column per line:
# * text
$ cat lm.tsv | head -1
I came back home because I had forgotten my wallet.
```

#### JSONL

```bash
# [standard generation] object with fields:
# * source_sequence (string)
# * target_sequence (string)
$ cat gen.jsonl | head -1
{"source_sequence": "I love these headphones!", "target_sequence": "Io adoro queste cuffie!"}

# [language modeling] object with fields:
# * source_sequence (string)
$ cat lm.jsonl | head -1
{"source_sequence": "I came back home because I had forgotten my wallet."}
```

:::tip

In several generation problems such as multilingual machine translation, depending on the chosen model, beside the source (and target) sequence,
you also want to specify the source (and target) language. To do so, just add the string fields *source_language* and *target_language* on each json object:

```bash
$ cat gen-with-langs.jsonl | head -1
{"source_sequence": "I love these headphones!", "source_language": "en", "target_sequence": "Io adoro queste cuffie!", "target_language": "it"}

$ cat lm-with-langs.jsonl | head -1
{"source_sequence": "I came back home because I had forgotten my wallet.", "source_language": "en"}
```

Note that:
* Language specification is not supported for `.tsv`
* Not all models support language specification (e.g. Bart), while others require it (e.g. mBart)

:::
