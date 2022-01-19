---
sidebar_position: 3
title: Default Profiles
---

As you have seen from the previous tutorials, your systems are fully customizable in `classy`.
Even if we strongly encourage you to create you own configurations, we provide a set of predefined and well-established profiles
that will work with competitive performances in almost all setting and scenarios.

:::tip
To use a profile, you just have to pass the profile name to the parameter `--profile` at training time
```bash
classy train <task> <dataset-path> -n <model-name> --profile <profile_name>
```
:::

##  distilbert :deciduous_tree: :rocket:

##### General Info

| Supported Tasks | Supported Languages | Required VRAM |
| 		:---:     |     :---:           |       :---:      |
| `sequence` `sentence-pair` `token` `qa` | English | < 4GB |


##### Model and Optimization

| Model | Optimizer |
| 		:---:     |     :---:           |
| <u>DistilBERT</u> (:page_facing_up: [Paper](https://arxiv.org/abs/1910.01108) \| :hammer: [Implementation](https://huggingface.co/transformers/model_doc/distilbert.html)) | <u>Adafactor</u> ([Paper](https://arxiv.org/abs/1804.04235) :page_facing_up: [Implementation](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#adafactor-pytorch) :hammer:) |

##### Train command
```bash
classy train [sequence|sentence-pair|token|qa] my_dataset_path -n my_model --profile distilbert
```
##### When should I use this profile?
- You want a **blazing fast** training and inference
- **Quick run** to evaluate your dataset and check for possible flaws
- You don't have at your disposal a GPU with more than 4GB VRAM
- You will use the model in **low energy consumption** scenarios

---

## distilroberta :deciduous_tree: :rocket:

##### General Info

| Supported Tasks | Supported Languages | Required VRAM |
| 		:---:     |     :---:           |       :---:      |
| `sequence` `sentence-pair` `token` `qa` | English | < 4GB |


##### Model and Optimization

| Model | Optimizer |
| 		:---:     |     :---:           |
| <u>DistilRoBERTa</u> (:hammer: [Implementation](https://huggingface.co/distilroberta-base)) | <u>Adafactor</u> ([Paper](https://arxiv.org/abs/1804.04235) :page_facing_up: [Implementation](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#adafactor-pytorch) :hammer:) |

##### Train command
```bash
classy train [sequence|sentence-pair|token|qa] my_dataset_path -n my_model --profile distilroberta
```
##### When should I use this profile?
- You want a **blazing fast** training and inference
- **Quick run** to evaluate your dataset and check for possible flaws
- You don't have at your disposal a GPU with more than 4GB VRAM
- You will use the model in **low energy consumption** scenarios

---

## squeezebert :deciduous_tree: :rocket:

##### General Info

| Supported Tasks | Supported Languages | Required VRAM |
| 		:---:     |     :---:           |       :---:      |
| `sequence` `sentence-pair` `token` `qa` | English | < 4GB |


##### Model and Optimization

| Model | Optimizer |
| 		:---:     |     :---:           |
| <u>SqueezeBERT</u> (:page_facing_up: [Paper](https://arxiv.org/abs/2006.11316) \| :hammer: [Implementation](https://huggingface.co/transformers/model_doc/squeezebert.html)) | <u>Adafactor</u> ([Paper](https://arxiv.org/abs/1804.04235) :page_facing_up: [Implementation](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#adafactor-pytorch) :hammer:) |

##### Train command
```bash
classy train [sequence|sentence-pair|token|qa] my_dataset_path -n my_model --profile squeezebert
```
##### When should I use this profile?
- You want a **blazing fast** training and inference
- **Quick run** to evaluate your dataset and check for possible flaws
- You don't have at your disposal a GPU with more than 4GB VRAM
- You will use the model in **low energy consumption** scenarios

---

## bert-base :evergreen_tree: :bullettrain_side:

##### General Info

| Supported Tasks | Supported Languages | Required VRAM |
| 		:---:     |     :---:           |       :---:      |
| `sequence` `sentence-pair` `token` `qa` | English | < 8GB |


##### Model and Optimization

| Model | Optimizer |
| 		:---:     |     :---:           |
| <u>BERT-base</u> (:page_facing_up: [Paper](https://aclanthology.org/N19-1423) \| :hammer: [Implementation](https://huggingface.co/transformers/model_doc/bert.html)) | <u>AdamW</u> ([Paper](https://arxiv.org/abs/1711.05101) :page_facing_up: [Implementation](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) :hammer:) |

##### Train command
```bash
classy train [sequence|sentence-pair|token|qa] my_dataset_path -n my_model --profile bert-base
```
##### When should I use this profile?
- You want a **trade-off between training/inference speed and model performances**
- You want a **well-established model** for everyday use
- You have at your disposal a GPU with at least 8GB of VRAM
- You will use the model in **moderate energy consumption** scenarios

---

## gpt2 :evergreen_tree: :bullettrain_side:

##### General Info

| Supported Tasks | Supported Languages | Required VRAM |
| 		:---:     |     :---:           |       :---:      |
| `generation` | English | < 8GB |


##### Model and Optimization

| Model | Optimizer |
| 		:---:     |     :---:           |
| <u>GPT2</u> (:page_facing_up: [Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) \| :hammer: [Implementation](https://huggingface.co/transformers/model_doc/gpt2.html)) | <u>Adam</u> ([Paper](https://arxiv.org/abs/1412.6980) :page_facing_up: [Implementation](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) :hammer:) |

##### Train command
```bash
classy train [generation] my_dataset_path -n my_model --profile gpt2
```
##### When should I use this profile?
- You want an affordable (decoder-only) **generative model** for English
- You have at your disposal a GPU with at least 8GB of VRAM
- You will use the model in **moderate energy consumption** scenarios

---

## roberta-base :evergreen_tree: :bullettrain_side:

##### General Info

| Supported Tasks | Supported Languages | Required VRAM |
| 		:---:     |     :---:           |       :---:      |
| `sequence` `sentence-pair` `token` `qa` | English | < 8GB |


##### Model and Optimization

| Model | Optimizer |
| 		:---:     |     :---:           |
| <u>RoBERTa-base</u> (:page_facing_up: [Paper](https://arxiv.org/abs/1907.11692) \| :hammer: [Implementation](https://huggingface.co/transformers/model_doc/bert.html)) | <u>AdamW</u> ([Paper](https://arxiv.org/abs/1711.05101) :page_facing_up: [Implementation](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) :hammer:) |

##### Train command
```bash
classy train [sequence|sentence-pair|token|qa] my_dataset_path -n my_model --profile bert-base
```
##### When should I use this profile?
- You want a **trade-off between training/inference speed and model performances**
- You want a **well-established model** for everyday use
- You have at your disposal a GPU with at least 8GB of VRAM
- You will use the model in **moderate energy consumption** scenarios

---

## deberta-base :evergreen_tree: :bullettrain_side:

##### General Info

| Supported Tasks | Supported Languages | Required VRAM |
| 		:---:     |     :---:           |       :---:      |
| `sequence` `sentence-pair` `token` `qa` | English | < 8GB |


##### Model and Optimization

| Model | Optimizer |
| 		:---:     |     :---:           |
| <u>DeBERTa-base</u> (:page_facing_up: [Paper](https://arxiv.org/abs/2006.03654) \| :hammer: [Implementation](https://huggingface.co/transformers/model_doc/deberta.html)) | <u>RAdam</u> ([Paper](https://arxiv.org/pdf/1908.03265v3.pdf) :page_facing_up: [Implementation](https://github.com/LiyuanLucasLiu/RAdam) :hammer:) |

##### Train command
```bash
classy train [sequence|sentence-pair|token|qa] my_dataset_path -n my_model --profile deberta-base
```
##### When should I use this profile?
- You want a **trade-off between training/inference speed and model performances**
- You want a **recently released model with state-of-the-art performances** on several NLU benchmarks
- You have at your disposal a GPU with at least 8GB of VRAM
- You will use the model in **moderate energy consumption** scenarios

---

## bart-base :evergreen_tree: :bullettrain_side:

##### General Info

| Supported Tasks | Supported Languages | Required VRAM |
| 		:---:     |     :---:           |       :---:      |
| `sequence` `sentence-pair` `token` `qa` `generation` | English | < 8GB |

##### Model and Optimization

| Model | Optimizer |
| 		:---:     |     :---:           |
| <u>Bart-base</u> (:page_facing_up: [Paper](https://arxiv.org/abs/1910.13461) \| :hammer: [Implementation](https://huggingface.co/transformers/model_doc/bart.html)) | <u>RAdam</u> ([Paper](https://arxiv.org/pdf/1908.03265v3.pdf) :page_facing_up: [Implementation](https://github.com/LiyuanLucasLiu/RAdam) :hammer:) |


##### Train command
```bash
classy train [sequence|sentence-pair|token|qa|generation] my_dataset_path -n my_model --profile bart-base
```

##### When should I use this profile?
- You want a **trade-off between training/inference speed and model performances**
- You want to tackle an English generation task with an affordable model
- You have at your disposal a GPU with at least 8GB of VRAM
- You will use the model in **moderate energy consumption** scenarios

---

## multilingual-bert :evergreen_tree: :bullettrain_side: :earth_asia:

##### General Info

| Supported Tasks | Supported Languages | Required VRAM |
| 		:---:     |     :---:           |       :---:      |
| `sequence` `sentence-pair` `token` `qa` | 104 ([Complete List](https://github.com/google-research/bert/blob/master/multilingual.md)) | < 8GB |


##### Model and Optimization

| Model | Optimizer |
| 		:---:     |     :---:           |
| <u>mBERT</u> (:page_facing_up: [Paper](https://aclanthology.org/N19-1423) \| :hammer: [Implementation](https://huggingface.co/transformers/model_doc/bert.html)) | <u>AdamW</u> ([Paper](https://arxiv.org/abs/1711.05101) :page_facing_up: [Implementation](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) :hammer:) |


##### Train command
```bash
classy train [sequence|sentence-pair|token|qa] my_dataset_path -n my_model --profile multilingual-bert
```
##### When should I use this profile?
- You require a **multilingual model** covering languages other than English
- You want a **trade-off between training/inference speed and model performances**
- You want a **well-established model** for everyday use
- You have at your disposal a GPU with at least 8GB of VRAM
- You will use the model in **moderate energy consumption** scenarios

---

## xlm-roberta-base :evergreen_tree: :bullettrain_side: :earth_asia:

##### General Info

| Supported Tasks | Supported Languages | Required VRAM |
| 		:---:     |     :---:           |       :---:      |
| `sequence` `sentence-pair` `token` `qa` | 100 (Complete list in the reference paper) | < 8GB |


##### Model and Optimization

| Model | Optimizer |
| 		:---:     |     :---:           |
| <u>XLM-RoBERTa-base</u> (:page_facing_up: [Paper](https://arxiv.org/abs/1911.02116) \| :hammer: [Implementation](https://huggingface.co/transformers/model_doc/xlmroberta.html)) | <u>AdamW</u> ([Paper](https://arxiv.org/abs/1711.05101) :page_facing_up: [Implementation](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) :hammer:) |


##### Train command
```bash
classy train [sequence|sentence-pair|token|qa] my_dataset_path -n my_model --profile xlm-roberta-base
```
##### When should I use this profile?
- You require a state-of-the-art **multilingual model** covering languages other than English
- You want a **trade-off between training/inference speed and model performances**
- You want a **well-established model** for everyday use
- You have at your disposal a GPU with at least 8GB of VRAM
- You will use the model in **moderate energy consumption** scenarios

---

## bert-large :cactus: :tractor:

##### General Info

| Supported Tasks | Supported Languages | Required VRAM |
| 		:---:     |     :---:           |       :---:      |
| `sequence` `sentence-pair` `token` `qa` | English | < 11GB (fp16) |


##### Model and Optimization

| Model | Optimizer |
| 		:---:     |     :---:           |
| <u>BERT-large</u> (:page_facing_up: [Paper](https://aclanthology.org/N19-1423) \| :hammer: [Implementation](https://huggingface.co/transformers/model_doc/bert.html)) | <u>AdamW</u> ([Paper](https://arxiv.org/abs/1711.05101) :page_facing_up: [Implementation](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) :hammer:) |

##### Train command
```bash
classy train [sequence|sentence-pair|token|qa] my_dataset_path -n my_model --profile bert-large --fp16
```
:::caution
Remember to use the `--fp16` at training time or otherwise the model may not fit in memory.
:::

##### When should I use this profile?
- You want state-of-the-art performances, **no compromise!**
- You want to show how far you can go with the proper infrastructure
- You want a **well-established model** used by thousands of users
- You have at your disposal a GPU with at least 11GB of VRAM that supports fp16 precision
- You don't have any energy consumption restriction

---

## roberta-large :cactus: :tractor:

##### General Info

| Supported Tasks | Supported Languages | Required VRAM |
| 		:---:     |     :---:           |       :---:      |
| `sequence` `sentence-pair` `token` `qa` | English | < 11GB (fp16) |

##### Model and Optimization

| Model | Optimizer |
| 		:---:     |     :---:           |
| <u>RoBERTa-large</u> (:page_facing_up: [Paper](https://arxiv.org/abs/1907.11692) \| :hammer: [Implementation](https://huggingface.co/transformers/model_doc/bert.html)) | <u>AdamW</u> ([Paper](https://arxiv.org/abs/1711.05101) :page_facing_up: [Implementation](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) :hammer:) |

##### Train command
```bash
classy train [sequence|sentence-pair|token|qa] my_dataset_path -n my_model --profile roberta-large --fp16
```
:::caution
Remember to use the `--fp16` at training time or otherwise the model may not fit in memory.
:::
##### When should I use this profile?
- You want state-of-the-art performances, **no compromise!**
- You want to show how far you can go with the proper infrastructure
- You want a **well-established model** used by thousands of users
- You have at your disposal a GPU with at least 11GB of VRAM that supports fp16 precision
- You don't have any energy consumption restriction

---

## deberta-large :cactus: :tractor:

##### General Info

| Supported Tasks | Supported Languages | Required VRAM |
| 		:---:     |     :---:           |       :---:      |
| `sequence` `sentence-pair` `token` `qa` | English | < 11GB (fp16) |

##### Model and Optimization

| Model | Optimizer |
| 		:---:     |     :---:           |
| <u>DeBERTa-large</u> (:page_facing_up: [Paper](https://arxiv.org/abs/2006.03654) \| :hammer: [Implementation](https://huggingface.co/transformers/model_doc/deberta.html)) | <u>RAdam</u> ([Paper](https://arxiv.org/pdf/1908.03265v3.pdf) :page_facing_up: [Implementation](https://github.com/LiyuanLucasLiu/RAdam) :hammer:) |

##### Train command
```bash
classy train [sequence|sentence-pair|token|qa] my_dataset_path -n my_model --profile deberta-large --fp16
```
:::caution
Remember to use the `--fp16` at training time or otherwise the model may not fit in memory.
:::
##### When should I use this profile?
- You want state-of-the-art performances, **no compromise!**
- You want to show how far you can go with the proper infrastructure
- You want **one of the latest released SotA models**
- You have at your disposal a GPU with at least 11GB of VRAM that supports fp16 precision
- You don't have any energy consumption restriction

---

## xlm-roberta-large :cactus: :tractor: :earth_asia:

##### General Info

| Supported Tasks | Supported Languages | Required VRAM |
| 		:---:     |     :---:           |       :---:      |
| `sequence` `sentence-pair` `token` `qa` | 100 (Complete list in the reference paper) | < 16GB (fp16) |

##### Model and Optimization

| Model | Optimizer |
| 		:---:     |     :---:           |
| <u>XLM-RoBERTa-large</u> (:page_facing_up: [Paper](https://arxiv.org/abs/1911.02116) \| :hammer: [Implementation](https://huggingface.co/transformers/model_doc/xlmroberta.html)) | <u>AdamW</u> ([Paper](https://arxiv.org/abs/1711.05101) :page_facing_up: [Implementation](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) :hammer:) |

##### Train command
```bash
classy train [sequence|sentence-pair|token|qa] my_dataset_path -n my_model --profile xlm-roberta-large --fp16
```
:::caution
Remember to use the `--fp16` at training time or otherwise the model may not fit in memory.
:::
##### When should I use this profile?
- You require a state-of-the-art **multilingual model** covering languages other than English, with **no compromise**
- You want to show how far you can go with the proper infrastructure
- You want a **well-established model** used by thousands of users
- You have at your disposal a GPU with at least 11GB of VRAM that supports fp16 precision
- You don't have any energy consumption restriction

---

## gpt2-medium :cactus: :tractor:

##### General Info

| Supported Tasks | Supported Languages | Required VRAM |
| 		:---:     |     :---:           |       :---:      |
| `generation` | English | < 11GB (fp16) |


##### Model and Optimization

| Model | Optimizer |
| 		:---:     |     :---:           |
| <u>GPT2</u> (:page_facing_up: [Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) \| :hammer: [Implementation](https://huggingface.co/transformers/model_doc/gpt2.html)) | <u>AdamW</u> ([Paper](https://arxiv.org/abs/1711.05101) :page_facing_up: [Implementation](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) :hammer:) |

##### Train command
```bash
classy train [generation] my_dataset_path -n my_model --profile gpt2-medium
```
##### When should I use this profile?
- You want a medium (decoder-only) **generative model** for English
- You have at your disposal a GPU with at least 11GB of VRAM that supports fp16 precision
- You don't have any energy consumption restriction

---

## bart-large :cactus: :tractor:

##### General Info

| Supported Tasks | Supported Languages | Required VRAM |
| 		:---:     |     :---:           |       :---:      |
| `sequence` `sentence-pair` `token` `qa` `generation` | English | < 11GB (fp16) |

##### Model and Optimization

| Model | Optimizer |
| 		:---:     |     :---:           |
| <u>Bart-large</u> (:page_facing_up: [Paper](https://arxiv.org/abs/1910.13461) \| :hammer: [Implementation](https://huggingface.co/transformers/model_doc/bart.html)) | <u>RAdam</u> ([Paper](https://arxiv.org/pdf/1908.03265v3.pdf) :page_facing_up: [Implementation](https://github.com/LiyuanLucasLiu/RAdam) :hammer:) |


##### Train command
```bash
classy train [sequence|sentence-pair|token|qa|generation] my_dataset_path -n my_model --profile bart-large
```

##### When should I use this profile?
- You want state-of-the-art performances, especially on English generation problems, with **no compromise!**
- You want to show how far you can go with the proper infrastructure
- You want a **well-established model** used by thousands of users
- You have at your disposal a GPU with at least 11GB of VRAM that supports fp16 precision
- You don't have any energy consumption restriction

---

## mbart :cactus: :building_construction: :earth_asia:

##### General Info

| Supported Tasks | Supported Languages | Required VRAM |
| 		:---:     |     :---:           |       :---:      |
| `sequence` `sentence-pair` `token` `qa` `generation` | English | < 24GB (fp16) |

##### Model and Optimization

| Model | Optimizer |
| 		:---:     |     :---:           |
| <u>mBART</u> (:page_facing_up: [Paper](https://arxiv.org/abs/2001.08210) \| :hammer: [Implementation](https://huggingface.co/transformers/model_doc/mbart.html)) | <u>RAdam</u> ([Paper](https://arxiv.org/pdf/1908.03265v3.pdf) :page_facing_up: [Implementation](https://github.com/LiyuanLucasLiu/RAdam) :hammer:) |


##### Train command
```bash
classy train [sequence|sentence-pair|token|qa] my_dataset_path -n my_model --profile bart-base
```

##### When should I use this profile?
- You want a state-of-the-art **multilingual model**, covering 25 languages and particularly suited for **generation tasks** (e.g. machine translation), with **no compromise**
- You want to show how far you can go with the proper infrastructure
- You want a **well-established model** used by thousands of users
- You have at your disposal a GPU with at least 24GB of VRAM that supports fp16 precision
- You don't have any energy consumption restriction

---

## gpt2-large :cactus: :building_construction:

##### General Info

| Supported Tasks | Supported Languages | Required VRAM |
| 		:---:     |     :---:           |       :---:      |
| `generation` | English | < 24GB (fp16) |


##### Model and Optimization

| Model | Optimizer |
| 		:---:     |     :---:           |
| <u>GPT2</u> (:page_facing_up: [Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) \| :hammer: [Implementation](https://huggingface.co/transformers/model_doc/gpt2.html)) | <u>AdamW</u> ([Paper](https://arxiv.org/abs/1711.05101) :page_facing_up: [Implementation](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) :hammer:) |

##### Train command
```bash
classy train [generation] my_dataset_path -n my_model --profile gpt2-large
```
##### When should I use this profile?
- You want a large (decoder-only) **generative model** for English
- You have at your disposal a GPU with at least 24GB of VRAM that supports fp16 precision
- You don't have any energy consumption restriction
