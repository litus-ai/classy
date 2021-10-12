---
sidebar_position: 1
title: Tasks
---

import ReactTermynal from '../../../src/components/termynal';

In its simplest flavor, classy lets **you focus entirely on your data** and automatically handles the development
of full-fledged models for you, with no ML knowledge or additional line of code required. To achieve this, classy only 
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

## Sequence Classification

Given a text in input (e.g. a sentence, a document), determine its most suitable label from a predefined set.

:::tip

An example is *Sentiment Analysis*:
* **input**: I love these headphones!
* **output**: positive

:::

## Sentence-Pair Classification

Given two texts in input (e.g. two sentences), determine the most suitable label for this pair (usually denoting some semantic relations) from a predefined set.

:::tip

An example is *Paraphrasis Detection*:
* **first input sentence**: I love these headphones!
* **second input sentence**: I really like these headphones!
* **output**: equivalent

:::

## Token Classification

Given a list of tokens, for each of them, determine its most suitable label from a predefined set.

:::tip

An example is *Part-of-Speech Tagging*:
* **input**: I love these headphones
* **output**: PRON VERB DET NOUN

:::

## Text Extraction

Given a context (e.g. a document) and some query about this document (e.g. a question), extract the consecutive span in the context that best addresses the query (e.g. an answer).

:::tip

An example is *Question Answering*:
* **input context**: I love these headphones!
* **input query**: What do you love?
* **output**: I love <mark style={{backgroundColor: "#f1e740"}}>these headphones</mark>!

:::

## Generation

Given a source text in input (e.g. a sentence), generate the corresponding target text.

:::tip

An example is *English-to-Italian Machine Translation*:
* **input**: I love these headphones!
* **output**: Io adoro queste cuffie!

:::
