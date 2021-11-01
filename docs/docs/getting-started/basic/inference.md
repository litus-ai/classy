---
sidebar_position: 5
title: Using your trained model
---

import ReactTermynal from '/src/components/termynal';

Now that we have our trained model called `fast-ner`, stored under `experiments/fast-ner/<date>/<time>`, we can use it!

`classy` offers a wide variety of commands to explore, test and deploy your trained models.

### Predicting

Use *fast-ner* to perform Named Entity Recognition on every sentence stored in a target file:

<ReactTermynal>
  <span data-ty="input">cat target.tsv | head -1</span>
  <span data-ty>Google 's headquarters are in California .</span>
  <span data-ty="input">classy predict file fast-ner target.tsv -o target.out.tsv</span>
  <span data-ty="progress"></span>
  <span data-ty>Prediction complete</span>
  <span data-ty="input">cat target.out.tsv | head -1</span>
  <span data-ty>Google 's headquarters are in California . &lt;tab&gt; ORG O O O O LOC O</span>
</ReactTermynal>

<p />

`classy predict` also supports an `interactive` mode. Check out [the documentation](/docs/reference-manual/cli/predict) for more details.
### Presenting

Present a demo of *fast-ner*:

<ReactTermynal>
  <span data-ty="input">classy demo fast-ner</span>
  <span data-ty data-ty-start-delay="2000">Demo up and running at http://0.0.0.0:8000</span>
</ReactTermynal>

<p />

The demo (available at http://localhost:8000/) has a page to try free-hand input texts or samples taken from a validation or test set, if available, 
and a page that shows the full configuration the model has been trained with. For more details, check out [`classy demo`'s documentation](/docs/reference-manual/cli/inference/#demo).

![Classy Demo](/img/intro/classy-demo-tok-model.png)

### Exposing via REST API

Expose *fast-ner* via a REST API that can be queried by any REST client:

<ReactTermynal>
  <span data-ty="input">classy serve fast-ner</span>
  <span data-ty data-ty-start-delay="2000">REST API up and running at http://0.0.0.0:8000</span>
  <span data-ty>Checkout the OpenAPI docs at http://localhost:8000/docs</span>
  <span data-ty="input">curl -X 'POST' 'http://localhost:8000/' -H 'accept: application/json' -H 'Content-Type: application/json' -d 
'[{'{'}"tokens": ["Google", "'\''s", "headquarters", "are", "in", "California", "."]{'}'}]'</span>
  <span data-ty data-ty-start-delay="2000">[{'{'}"tokens": ["Google", "'s", "headquarters", "are", "in", "California", "."], "labels": ["ORG", "O", "O", "O", "O", "LOC", "O"]{'}'}]</span>
</ReactTermynal>

<p />

We also automatically generate the OpenAPI documentation page!

![Classy Serve Docs](/img/intro/classy-serve-tok.png)
