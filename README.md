# Detecting Voice Clones in Indian-Language Phone Calls and Voice Notes

**Status: work in progress.** A first Hindi detector is trained and checked, and its main weakness is found: one unfamiliar text-to-speech system gets past it. Robustness training and a demo come next (last updated 9 October 2026). Hindi only so far; Telugu and Tamil come later.

Voice-clone scams reach people through phone calls and WhatsApp voice notes, where audio is compressed and noisy. Most fake-voice detectors are built and tested on clean English or Chinese recordings. This project measures how an existing detector holds up on Indian-language audio under phone-style conditions, then trains one that copes better.

## Headline

| Detector | EER, clean Hindi test set | EER, worst phone condition (8 kHz GSM) |
|---|---|---|
| AASIST, trained on English research data | 73.4% | 55.0% |
| This project's detector, trained on Hindi | 0.0% | 0.7% |

EER is the Equal Error Rate: 0% is perfect, 50% is a coin flip. Both are scored on the same 696 held-out clips from speakers never seen in training. Details, checks and limits are below.

## Baseline: an English-trained detector

AASIST, a standard detector trained on English research data, was scored on a clean Hindi test set. The metric is Equal Error Rate (EER): 0% is perfect, 50% is a coin flip.

| Test set | EER | 95% range | Clips (fake / real) |
|---|---|---|---|
| English data AASIST was built for (ASVspoof 2019 LA, sample) | 1.0% | 0.0 to 2.7 | 200 / 200 |
| Clean Hindi, all fakes | 73.4% | 70.5 to 77.4 | 371 / 325 |
| Clean Hindi, `freevc24` fakes | 89.7% | 87.2 to 92.4 | 229 / 325 |
| Clean Hindi, `xtts_v2` fakes | 55.8% | 51.7 to 61.4 | 142 / 325 |

Using the detector's own real/fake decision on the Hindi test set:

- 90.2% of real clips were called fake.
- 88.2% of `freevc24` fakes were passed as real.
- 19.7% of `xtts_v2` fakes were passed as real.

So a detector that is near-perfect on its home data is worse than a coin flip on Hindi, before any phone compression is added. An EER well above 50% means its scores run the wrong way round: it rates the fakes as more real than the real speech.

### Under voice-note and phone-call compression

Every test clip, real and fake, was encoded the way a voice note or phone call would encode it, decoded back, and scored again.

| Condition | EER, all fakes | EER, `freevc24` | EER, `xtts_v2` | Real called fake | `freevc24` passed as real | `xtts_v2` passed as real |
|---|---|---|---|---|---|---|
| Clean | 73.4% | 89.7% | 55.8% | 90.2% | 88.2% | 19.7% |
| Voice note, Opus 24 kbps | 66.5% | 83.8% | 45.8% | 85.2% | 81.7% | 13.4% |
| Voice note, Opus 12 kbps | 61.8% | 80.0% | 46.5% | 89.8% | 75.1% | 7.7% |
| Phone call, 8 kHz G.711 | 70.0% | 85.7% | 47.4% | 97.2% | 73.4% | 7.7% |
| Phone call, 8 kHz GSM | 55.0% | 68.7% | 32.4% | 97.5% | 27.5% | 1.4% |

The more the audio is degraded, the more AASIST calls everything fake. On a GSM call it flags 97.5% of real speakers as fake, and its overall error rate stays between 55% and 73% in every condition. This fits the idea that it reacts to recording quality more than to whether a voice is cloned, but it does not prove it.

Two limits on this test:

- **Mobile codec.** Most mobile calls use AMR, which the ffmpeg build on Colab cannot encode. GSM full-rate is used as the nearest available stand-in.
- **No background noise yet.** These conditions change the codec and bandwidth only.

### What this does and does not show

- **The test code is right.** The same loading and scoring code gives 1.0% on the English set, against the 0.83% published for this model on the full set.
- **The cause is not known yet.** The Hindi set differs from AASIST's training data in language, recording conditions (phone recordings against studio recordings) and cloning tools, all at once. This test cannot say which of those is responsible.
- **The ranges are optimistic.** They come from re-drawing test clips, not test speakers, and the test set has only about 15 speakers.

## A detector trained on Hindi

A pretrained multilingual speech model, XLS-R (300M parameters, trained on 128 languages), turns each clip into features. It is not retrained. A small classifier on top learns real against fake from the 4,158 clean Hindi training clips: it weighs the model's 25 layers and draws one line between the classes. Validation EER was 0.0% from the first epoch.

### Clean and degraded test audio

Trained on clean audio only. Decisions in the last two columns use one fixed threshold, set on clean validation clips.

| Condition | EER, all fakes | EER, `freevc24` | EER, `xtts_v2` | Real called fake | Fakes passed as real | AASIST EER |
|---|---|---|---|---|---|---|
| Clean | 0.0% | 0.0% | 0.0% | 0.6% | 0.0% | 73.4% |
| Voice note, Opus 24 kbps | 0.0% | 0.0% | 0.0% | 0.9% | 0.0% | 66.5% |
| Voice note, Opus 12 kbps | 0.0% | 0.0% | 0.0% | 4.9% | 0.0% | 61.8% |
| Phone call, 8 kHz G.711 | 0.3% | 0.4% | 0.0% | 1.5% | 0.0% | 70.0% |
| Phone call, 8 kHz GSM | 0.7% | 0.4% | 1.5% | 13.8% | 0.0% | 55.0% |

The ranking of real against fake barely moves under compression. The fixed threshold does drift: on a GSM call, 13.8% of real speakers are flagged as fake. Training with compressed audio is meant to fix that.

### A cloning tool it never saw

Clean audio, with one tool left out of training entirely.

| Trained on | EER, `freevc24` | EER, `xtts_v2` |
|---|---|---|
| Both tools | 0.0% | 0.0% |
| `freevc24` only | 0.0% | 5.9% (unseen) |
| `xtts_v2` only | 1.8% (unseen) | 0.0% |

### Checking for a shortcut

A score this good needs checking. Every real clip comes from one source (Kathbath phone recordings) and every fake from another (IndicSynth), so a detector could learn where a recording came from instead of whether the voice is cloned. Two checks argue against that.

**1. Basic audio measurements cannot do it.** Seven plain measurements with no AI (background level, loudness range, brightness, spectral roll-off, noisiness, share of energy above 4 kHz, zero crossings), fed to a logistic regression, get 35.2% EER on clean audio and 49% to 57% under compression. There is no obvious recording-quality giveaway in the data.

| Condition | 7 basic measurements, EER | Hindi detector, EER |
|---|---|---|
| Clean | 35.2% | 0.0% |
| Voice note, Opus 24 kbps | 49.3% | 0.0% |
| Voice note, Opus 12 kbps | 49.9% | 0.0% |
| Phone call, 8 kHz G.711 | 56.3% | 0.3% |
| Phone call, 8 kHz GSM | 57.3% | 0.7% |

**2. It does not call real speech from other sources fake.** Real Hindi speech from two sources the detector never saw, scored with the same fixed threshold:

| Real speech from | Clips | Called fake | EER against the test fakes |
|---|---|---|---|
| Kathbath test set (same source as training) | 325 | 0.6% | 0.0% |
| FLEURS (people reading on their own devices) | 300 | 0.7% | 0.0% |
| IndicTTS (studio recordings) | 300 | 1.7% | 0.0% |

If the detector had learned "phone recording means real", it would have called the clean studio recordings fake. It called 1.7% of them fake.

### Fakes from systems it never saw

Every fake above comes from IndicSynth's two tools. To test fakes from elsewhere, 150 Hindi test-set sentences were spoken by each of two text-to-speech systems the detector never saw: MMS-TTS Hindi (a VITS model, one voice) and SeamlessM4T v2 (10 of its voices). These are stock voices, not clones of a particular person. Both were scored against the 325 real test clips, clean and after the same encodings.

EER (lower is better):

| Fakes from | Clean | Opus 24k | Opus 12k | G.711 | GSM |
|---|---|---|---|---|---|
| IndicSynth test fakes (tools seen in training) | 0.0% | 0.0% | 0.0% | 0.3% | 0.7% |
| SeamlessM4T v2 (unseen) | 0.2% | 0.8% | 1.4% | 0.5% | 1.9% |
| MMS-TTS Hindi (unseen) | 4.0% | 7.4% | 11.4% | 10.4% | 16.3% |

Fakes passed as real at the detector's fixed threshold (lower is better):

| Fakes from | Clean | Opus 24k | Opus 12k | G.711 | GSM |
|---|---|---|---|---|---|
| IndicSynth test fakes | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| SeamlessM4T v2 | 0.0% | 0.7% | 0.0% | 0.7% | 0.0% |
| MMS-TTS Hindi | 36.7% | 48.7% | 30.0% | 60.7% | 18.0% |

SeamlessM4T is caught almost as well as the training tools. MMS-TTS is the weak spot: the detector still ranks most of its clips as less real than real speech (4% EER on clean audio), but at its working threshold it lets 30% to 61% of them through. Compression makes it worse. MMS-TTS uses one voice and 150 clips, so these numbers describe one system and voice, not text-to-speech in general.

### What this does and does not show

- **Real-speech side: holds up.** It recognises real Hindi speech from three different recording setups.
- **Fake side: mixed.** Leaving one IndicSynth tool out costs up to 5.9% EER, and SeamlessM4T v2 is caught almost perfectly, but MMS-TTS gets past the fixed threshold 30% to 61% of the time. Commercial cloning services have not been tested.
- **No background noise yet.** The degraded conditions change the codec and bandwidth only.
- **Small test set.** About 15 test speakers, so small differences between rows are not meaningful.

## Dataset

5,766 Hindi clips, all converted to 16 kHz mono WAV.

| | Clips | Voices | Source |
|---|---|---|---|
| Real speech | 3,000 | 97 | Kathbath (IndicSUPERB) |
| Fake, `freevc24` (voice conversion) | 1,500 | 101 | IndicSynth |
| Fake, `xtts_v2` (text to speech) | 1,266 | 46 | IndicSynth |

| Split | Real | `freevc24` | `xtts_v2` |
|---|---|---|---|
| Train | 2,195 | 1,044 | 919 |
| Validation | 480 | 227 | 205 |
| Test | 325 | 229 | 142 |

The fakes are clones of the same speakers who appear in the real clips, and the split is by speaker, so no voice is in more than one split.

Three differences between real and fake clips were removed, because a detector could use them to score well without learning anything about fakes:

- **Format and sample rate.** Real clips were 16 kHz FLAC and fakes were 24 kHz WAV. Everything is converted to 16 kHz WAV.
- **Loudness.** Fakes averaged about -16 dB and real clips about -24 dB. Every clip is set to -23 dB when it is loaded.
- **Speaker variety.** `xtts_v2` first had only 12 voices. It was resampled to 46 voices with an equal number of clips from each.

No audio is stored in this repository. The notebook downloads it from the original sources.

## Run it

Open the notebooks in Google Colab and run the cells in order:

1. [`notebooks/01_dataset_and_baseline.ipynb`](notebooks/01_dataset_and_baseline.ipynb) builds the dataset, saves it to Google Drive and runs the baseline test.
2. [`notebooks/02_compression_test.ipynb`](notebooks/02_compression_test.ipynb) restores the dataset from Drive and runs the compression test. Use a GPU runtime.
3. [`notebooks/03_detector_and_shortcut_check.ipynb`](notebooks/03_detector_and_shortcut_check.ipynb) runs end to end with Run all: setup, the compression test, training the Hindi detector, and the shortcut checks. About 25 minutes on a T4 GPU the first time.
4. [`notebooks/04_outside_fakes.ipynb`](notebooks/04_outside_fakes.ipynb) generates fakes with MMS-TTS and SeamlessM4T v2 and scores them. About 20 minutes on a T4 GPU the first time.
5. [`notebooks/05_robust_training.ipynb`](notebooks/05_robust_training.ipynb) trains three versions of the detector (clean audio; plus noise and phone audio; plus extra fakes) and compares them. About 45 minutes on a T4 GPU the first time.

The web demo is in [`demo/`](demo/): a Streamlit app that runs on Streamlit Community Cloud's free CPU tier. Upload or record a Hindi clip, including a WhatsApp voice note, and it says whether it sounds real or AI-generated. It needs a trained classifier file (`xlsr_head_clean.pt`, saved to Google Drive by notebook 03) in the `demo` folder. Deployment steps are in [`demo/README.md`](demo/README.md).

For the first notebook you need a free Hugging Face account and must accept the terms on the [Kathbath dataset page](https://huggingface.co/datasets/ai4bharat/Kathbath) first. The run downloads about 20 GB in pieces and deletes each piece after use. The baseline test takes about 10 minutes on a CPU.

## Plan

1. **Set up the data.** Done.
2. **Measure the baseline on clean audio.** Done.
3. **Simulate the real world.** Re-encode the test clips like voice notes (low-bitrate Opus) and phone calls (8 kHz) and score them again. Done for codecs; background noise is still to come.
4. **Fix it.** A detector trained on Hindi is done, including tests on an unseen cloning tool and unseen real-speech sources. Tested against fakes from two outside systems: one is caught, one (MMS-TTS) often gets through. Still to do: train with more varied fakes and with compressed and noisy audio, then re-test on a system kept out of training.
5. **Ship it.** A small web demo, plus Telugu and Tamil.

## Data and credits

- **IndicSynth** (Sharma, Ekbote and Gupta, ACL 2025): synthetic speech in 12 Indian languages. Licence CC BY-NC 4.0, so this project is non-commercial. [Dataset](https://huggingface.co/datasets/vdivyasharma/IndicSynth)
- **Kathbath / IndicSUPERB** (Javed et al., AAAI 2023): real speech in 12 Indian languages, from AI4Bharat. [Dataset](https://huggingface.co/datasets/ai4bharat/Kathbath), [code](https://github.com/AI4Bharat/IndicSUPERB)
- **AASIST** (Jung et al., ICASSP 2022): the baseline detector and its pretrained weights, MIT licence. [Code](https://github.com/clovaai/aasist)
- **XLS-R** (Babu et al., 2021): the pretrained multilingual speech model, `facebook/wav2vec2-xls-r-300m`, Apache 2.0 licence. [Model](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
- **FLEURS** (Conneau et al., 2022): real Hindi read speech, used only for testing, CC BY 4.0. [Dataset](https://huggingface.co/datasets/google/fleurs)
- **IndicTTS Hindi** (IIT Madras): studio recordings of real Hindi speech, used only for testing, through [a copy on Hugging Face](https://huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi).
- **MMS-TTS Hindi** (Pratap et al., 2023): `facebook/mms-tts-hin`, used to generate test fakes, CC BY-NC 4.0. [Model](https://huggingface.co/facebook/mms-tts-hin)
- **SeamlessM4T v2** (Seamless Communication et al., 2023): `facebook/seamless-m4t-v2-large`, used to generate test fakes, CC BY-NC 4.0. [Model](https://huggingface.co/facebook/seamless-m4t-v2-large)
- **ASVspoof 2019 LA**: English test set, used only to check the test code, through [a copy on Hugging Face](https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA). [Project site](https://www.asvspoof.org/)

## Responsible use

This project only detects cloned voices. It uses existing research datasets and does not generate clones of anyone.
