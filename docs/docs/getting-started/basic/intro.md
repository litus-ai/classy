---
sidebar_position: 1
title: First steps
---

In the following tutorial, we are going to go over a standard `classy` example, covering **Token Classification**, specifically
*Named Entity Recognition*, and you are going to learn the basics of how to use `classy` without (potentially) writing a single line of code.

## The Task

*Named Entity Recognition* is the task of identifying Named Entities (from a predefined set, e.g., **ORG**, **LOC** and **PER**) in a text.
For instance, consider what will be our running example for this whole tutorial:


<table>
<tr style={{textAlign: 'center'}}>
    <td>Barack</td>
    <td>Obama</td>
    <td>visited</td>
    <td>Google</td>
    <td>in</td>
    <td>California</td>
</tr>
<tr style={{textAlign: 'center'}}>
    <td>PER</td>
    <td>PER</td>
    <td>O</td>
    <td>ORG</td>
    <td>O</td>
    <td>LOC</td>
</tr>
</table>

Our goal is to train a classification model that, given a sequence of tokens as input,
outputs a sequence of Named Entity tags corresponding to each token in the sequence.


:::info
The contents of this tutorial focus on the task of *Token Classification*, but the general ideas and code structure are
very similar for other tasks as well (you can check them out [in the documentation](/docs/reference-manual/tasks-and-formats)).
:::
