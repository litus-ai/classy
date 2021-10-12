---
sidebar_position: 2
title: Input Formats
---

import ReactTermynal from '../../../src/components/termynal';

Once you have realized which task you are dealing with, you have to organize your data. Currently, for each of 
these tasks, classy supports two possible input formats:
* .tsv
* .jsonl

:::caution

classy automatically infers how to process a given file from its extension. So please explicitly use ".tsv" for .tsv files
and ".jsonl" for .jsonl files.

:::

## .tsv

With the .tsv format, files are expected to be standard .tsv files. Their structure varies depending on the selected task:

### Sequence Classification

```bash
# two columns per line:
# * input text 
# * corresponding label
$ cat seq.tsv | head -1
I love these headphones!    <tab>   positive
```

### Sentence-Pair Classification

```bash
# three columns per line:
# * first input text
# * second input text
# * label
$ cat sent-p.tsv | head -1
I love these headphones!    <tab>   I love these headphones!    <tab>   equivalent  
```

### Token Classification

```bash
# two columns per line:
# * space-separated list of input tokens
# * space-separated list of labels
$ cat tok.tsv | head -1
I love these headphones !   <tab>   PRON VERB DET NOUN PUNCT
```

### Text Extraction

```bash
# four columns per line:
# * context
# * query
# * start char offset
# * end char offset
$ cat text-ext.tsv | head -1
I love these headphones!    <tab>   What do you love?           <tab>   7  <tab>  23
```

### Generation

For generation, the input format depends on whether you wish to do standard generation or language modeling:
* **standard generation**: given a source text (e.g. *I loved these headphones!*), generate the corresponding target text (e.g. *Io adoro queste cuffie!*)
* **language modeling**: given a text (e.g. *I came back home because*), generate the subsequent words (e.g. *I had forgotten my wallet*)

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

## .jsonl

With the .jsonl format, each line contains a JSON object that represents a sample, whose type depends on the task under 
consideration:

### Sequence Classification

```bash
# object with fields: 
# * sequence (string)
# * label (string)
$ cat seq.jsonl | head -1
{"sequence": "I love these headphones!", "label": "positive"}   
```

### Sentence-Pair Classification

```bash
# object with fields: 
# * sentence1 (string)
# * sentence2 (string)
# * label (string)   
$ cat sent-p.jsonl | head -1
{"sentence1": "I love these headphones!", "sentence2": "I love these headphones!", "label": "equivalent"}  
```

### Token Classification

```bash
# object with fields: 
# * tokens (list of strings)
# * labels (list of strings)
$ cat tok.jsonl | head -1
{"tokens": ["I", "love", "these", "headphones", "!"], "labels": ["PRON", "VERB", "DET", "NOUN", "PUNCT"]}
```

### Text Extraction

```bash
# object with fields: 
# * context (string)
# * query (string)
# * answer_start (int) as a char offset
# * answer_end (int) as a char offset
$ cat text-ext.jsonl | head -1
{"context": "I love these headphones!", "query": "What do you love?", "answer_start", 7, "answer_end": 23}
```

### Generation

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
* Language specification is not supported for .tsv
* Not all models support language specification (e.g. Bart), while others require it (e.g. mBart)

:::