---
id: token-batch-size
title: Token Batch Size
hoverText: The (approximate) number of tokens (or BPEs) that can be in a batch.
---

### Term explanation

Token batching is a custom batching technique that relies on counting the number of input elements in a batch and
constrains them in order to have batches that don't overflow the available VRAM. For example, if our `token batch size`
were set at 1000, a batch of size (100, 10) would be acceptable, as well as one of size (20, 50), but not (50, 50) (as `50 x 50 = 2500`).
