---
sidebar_position: 3
title: evaluate/demo/serve
---

import ReactTermynal from '/src/components/termynal';

## Evaluate

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
unless it was also moved (and placed in a symmetric location in the file-system). Should it fail, providing the
path explicitly should solve the issue.

:::

## Serve

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

![Classy Serve Docs](/img/intro/classy-serve.png)

:::tip

By default, `classy serve` uses port 8000. Use the *-p* parameter to specify a different one.

:::

## Demo

You can use `classy demo` to spawn a [Streamlit](https://streamlit.io/) demo of your model.

<ReactTermynal>
  <span data-ty="input">classy demo sequence-example</span>
  <span data-ty data-ty-start-delay="2000">Demo up and running at http://0.0.0.0:8000</span>
</ReactTermynal>

<p />

Now you can check out out the demo at *http://0.0.0.0:8000*!

![Classy Demo - Model](/img/intro/classy-demo-seq-model.png)
![Classy Demo - Config](/img/intro/classy-demo-seq-config.png)
