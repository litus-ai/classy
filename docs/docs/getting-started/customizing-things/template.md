---
sidebar_position: 1
title: Classy Template
---

Welcome! If you're reading this, it's probably because you want to do something that `classy` does not support natively.

We have prepared a [GitHub template](https://github.com/sunglasses-ai/classy-template) that you can use to start things off without having to worry about folder structure and so on.

Our template is built to seamlessly integrate with `classy`, for instance:
- Any python module under `src/` is automatically imported when running any `classy` command. This way, you can, for example,
[register a new custom data format](/docs/getting-started/customizing-things/custom-data-format/) and start using it right away without needing to explicitly set it inside your configuration files.
- Any `.yaml` file under `configurations` is automatically discovered by [`hydra`](https://hydra.cc/) and can be used interchangeably with the configurations provided by `classy` itself.

  This is particularly useful when dealing with complex models / profiles (which we cover in the next section) which cannot fit inside `classy`'s main package.


:::tip

We have also prepared a [repository of examples](https://github.com/sunglasses-ai/classy-examples) that are
based on the template and on the contents of these guides to help you move your first steps with `classy`'s advanced features.

:::
