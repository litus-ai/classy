---
sidebar_position: 2
title: predict
---

import ReactTermynal from '/src/components/termynal';


You can use `classy predict` to perform predictions with a trained model. Two modes are supported:
* File-based prediction
* Bash-interactive

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
  <span data-ty>I wish I had never bought these terrible headphones! &lt;tab&gt; negative</span>
</ReactTermynal>

<p />

On the other hand, bash-interactive predictions allows you to interactively query models via bash:

<ReactTermynal>
  <span data-ty="input">classy predict interactive sequence-sample</span>
  <span data-ty="input" data-ty-prompt="Enter sequence text: ">I wish I had never bought these terrible headphones!</span>
  <span data-ty data-ty-start-delay="2000">  # prediction: negative</span>
  <span data-ty data-ty-prompt="Enter sequence text: "></span>
</ReactTermynal>

<p />
