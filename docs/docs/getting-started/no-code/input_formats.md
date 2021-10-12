---
sidebar_position: 2
title: Input Formats
---

import ReactTermynal from '../../../src/components/termynal';

Once you have realized which of these tasks you are dealing with, you have to organize your data. Currently, for each of 
these tasks, classy supports two possible input formats:
* .tsv
* .jsonl

:::caution

classy automatically infers how to process a given file from its extension. So please explicitly use ".tsv" for .tsv files
and ".jsonl" for .jsonl files.

:::

## .tsv

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

## .jsonl

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

