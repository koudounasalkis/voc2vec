# voc2vec: A Foundation Model for Non-Verbal Vocalization

This repository contains the code for the paper "**voc2vec: A Foundation Model for Non-Verbal Vocalization**", accepted at ICASSP 2025.

[![voc2vec-hubert-LS](https://img.shields.io/badge/voc2vec_hubert_LS-HuggingFace-blue)](https://huggingface.co/alkiskoudounas/voc2vec-hubert-ls-pt)
[![voc2vec-LS](https://img.shields.io/badge/voc2vec_LS-HuggingFace-red)](https://huggingface.co/alkiskoudounas/voc2vec-ls-pt)
[![voc2vec-AS](https://img.shields.io/badge/voc2vec_AS-HuggingFace-green)](https://huggingface.co/alkiskoudounas/voc2vec-as-pt)
[![voc2vec](https://img.shields.io/badge/voc2vec-HuggingFace-yellow)](https://huggingface.co/alkiskoudounas/voc2vec)

We propose a novel foundation model, **voc2vec**, specifically designed for non-verbal human data leveraging exclusively open-source non-verbal audio datasets. We employ a collection of 10 datasets covering around 125 hours of non-verbal audio.

Experimental results prove that voc2vec is effective in non-verbal vocalization classification, and it outperforms conventional speech and audio foundation models. Moreover, voc2vec consistently outperforms strong baselines, OpenSmile, and emotion2vec, on six different benchmark datasets. 

voc2vec is the **first universal representation model for vocalization tasks**.

> [!IMPORTANT]  
**04/14/2025.** We released a new model, [voc2vec-hubert-ls-pt](https://huggingface.co/alkiskoudounas/voc2vec-hubert-ls-pt), that continues pre-training from a HuBERT-like model originally pre-trained on LibriSpeech.
This model currently achieves state-of-the-art results (see the [Results section](#results)).

## Table of Contents

- [Pretraining](#pretraining)
- [Finetuning](#finetuning)
- [Results](#results)
- [Usage](#usage)
- [Models](#models)
- [Citation](#citation)
- [License](#license)

## Pretraining

The core contribution of voc2vec lies in the careful selection of diverse, open-source datasets for pre-training, specifically chosen to focus on non-verbal vocalizations. 
These datasets collectively cover around 125 hours of audio, ensuring that the model is exposed to a wide variety of human vocalizations, typically underrepresented in speech datasets.
Each dataset is chosen to represent different forms of non-verbal communication, such as emotional bursts, human reactions, and environmental sounds that involve vocal interaction. 
The datasets used for pre-training are summarized in the table below. 

| Dataset                                 | Dur. (h) | \# Samples | Avg Dur. (s) |
|-----------------------------------------|:--------:|:----------:|:------------:|
| AudioSet (vocalization)                 |   36.94  |    13439   |     9.90     |
| FreeSound (babies)                      |   23.42  |    1450    |     58.15    |
| HumanVoiceDataset                       |   0.06   |     179    |     1.21     |
| NNIME                                   |   3.55   |    5596    |     2.28     |
| NonSpeech7K                             |   6.72   |    6983    |     3.46     |
| ReCANVo                                 |   2.46   |    7077    |     1.25     |
| SingingDatabase                         |   3.97   |     113    |    126.48    |
| TUT (babies)                            |   13.17  |    1540    |     30.79    |
| VocalSketch                             |   10.53  |    10705   |     3.54     |
| VocalSound                              |   24.37  |    20985   |     4.18     |
| **Voc125 (Total)**                      |**125.19**| **68067**  |   **6.67**   |


## Finetuning

We evaluate voc2vec on six classification tasks using diverse datasets, each covering different types of non-verbal vocalizations. 
The datasets and their characteristics are summarized in the table below.

| Dataset                       | \# Classes | Dur. (h) | \# Samples | \# Avg Dur. (s) |
|-------------------------------|------------|:--------:|:----------:|:---------------:|
| ASVP-ESD                      |     13     |   15.07  |    12625   |       4.30      |
| ASVP-ESD (babies)             |      7     |   2.91   |    1339    |       8.22      |
| CNVVE                         |      6     |    0.2   |     921    |       0.78      |
| Donate A Cry                  |      5     |   0.88   |     457    |       6.93      |
| NonVerbal Vocalization        |     16     |    0.6   |     800    |       3.10      |
| VIVAE                         |      6     |   0.27   |    1085    |       0.90      |

## Results 

Here are the results of the voc2vec collection on the six datasets mentioned above:

| Model | Architecture | Pre-training DS | UAR | F1 Macro |
|--------|-------------|-------------|-----------|-----------|
| **voc2vec** | wav2vec 2.0 | Voc125 | .612±.212 | .580±.230 |
| **voc2vec-as-pt** | wav2vec 2.0 | AudioSet + Voc125 | .603±.183 | .574±.194 |
| **voc2vec-ls-pt** | wav2vec 2.0 | LibriSpeech + Voc125 | .661±.206 | .636±.223 |
| **voc2vec-hubert-ls-pt** | HuBERT | LibriSpeech + Voc125 | **.696±.189** | **.678±.200** |
| **wav2vec2-ls** | wav2vec 2.0 | LibriSpeech | .599±.237 | .569±.259 |
| **hubert-ls** | HuBERT | LibriSpeech | .627±.214 | .611±.222 |

## Usage

The model can be loaded using the `transformers` library. You need to install the following dependencies:

```bash
pip install transformers
pip install librosa
```

Then, you can load and use the model as follows:

```python
import torch
import librosa
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

## Load an audio file
audio_array, sr = librosa.load("path_to_audio.wav", sr=16000)

## Load model and feature extractor
MODEL_NAME = "alkiskoudounas/voc2vec-hubert-ls" # alkiskoudounas/voc2vec, alkiskoudounas/voc2vec-as-pt, alkiskoudounas/voc2vec-ls-pt, alkiskoudounas/voc2vec-hubert-ls
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)   
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME) 

## Extract features
inputs = feature_extractor(audio_array.squeeze(), sampling_rate=feature_extractor.sampling_rate, padding=True, return_tensors="pt")

## Compute logits
logits = model(**inputs).logits
```


## Models

We open-source three models:

| Model | Description | Link |
|--------|-------------|------|
| **voc2vec** | Pre-trained model on **125 hours of non-verbal audio**. | [🔗 Model](https://huggingface.co/alkiskoudounas/voc2vec) |
| **voc2vec-as-pt** | Continues pre-training from a wav2vec2-like model that was **initially trained on the AudioSet dataset**. | [🔗 Model](https://huggingface.co/alkiskoudounas/voc2vec-as-pt) |
| **voc2vec-ls-pt** | Continues pre-training from a wav2vec2-like model that was **initially trained on the LibriSpeech dataset**. | [🔗 Model](https://huggingface.co/alkiskoudounas/voc2vec-ls-pt) |
| **voc2vec-hubert-ls-pt** | Continues pre-training from a hubert-like model that was **initially trained on the LibriSpeech dataset**. | [🔗 Model](https://huggingface.co/alkiskoudounas/voc2vec-hubert-ls-pt) |

For more information about the model, please refer to the [paper](https://ieeexplore.ieee.org/abstract/document/10890672).

## Citation

> [!IMPORTANT]  
If you use this model in your research, please cite the following paper:

```bibtex
@inproceedings{koudounas2025voc2vec,
  title={voc2vec: A Foundation Model for Non-Verbal Vocalization},
  author={Koudounas, Alkis and La Quatra, Moreno and Siniscalchi, Sabato Marco and Baralis, Elena},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

## License

This code and the models are released under the Apache 2.0 license. See the [LICENSE](LICENSE) file for more details.
