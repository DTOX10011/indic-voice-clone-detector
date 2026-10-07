# Detecting Voice Clones in Indian-Language Phone Calls and Voice Notes

**Status: work in progress.** Phases 1 and 2 of 5 are done and Phase 3 is under way (last updated 7 October 2026). Hindi only so far; Telugu and Tamil come later.

Voice-clone scams reach people through phone calls and WhatsApp voice notes, where audio is compressed and noisy. Most fake-voice detectors are built and tested on clean English or Chinese recordings. This project measures how an existing detector holds up on Indian-language audio under phone-style conditions, then trains one that copes better.

## Result so far

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

For the first notebook you need a free Hugging Face account and must accept the terms on the [Kathbath dataset page](https://huggingface.co/datasets/ai4bharat/Kathbath) first. The run downloads about 20 GB in pieces and deletes each piece after use. The baseline test takes about 10 minutes on a CPU.

## Plan

1. **Set up the data.** Done.
2. **Measure the baseline on clean audio.** Done.
3. **Simulate the real world.** Re-encode the test clips like voice notes (low-bitrate Opus) and phone calls (8 kHz) and score them again: done for the baseline detector. Background noise is still to come.
4. **Fix it.** Retrain a detector with compressed and noisy audio in the training data, and test it on a cloning tool or compression setting it never saw.
5. **Ship it.** A small web demo, plus Telugu and Tamil.

## Data and credits

- **IndicSynth** (Sharma, Ekbote and Gupta, ACL 2025): synthetic speech in 12 Indian languages. Licence CC BY-NC 4.0, so this project is non-commercial. [Dataset](https://huggingface.co/datasets/vdivyasharma/IndicSynth)
- **Kathbath / IndicSUPERB** (Javed et al., AAAI 2023): real speech in 12 Indian languages, from AI4Bharat. [Dataset](https://huggingface.co/datasets/ai4bharat/Kathbath), [code](https://github.com/AI4Bharat/IndicSUPERB)
- **AASIST** (Jung et al., ICASSP 2022): the baseline detector and its pretrained weights, MIT licence. [Code](https://github.com/clovaai/aasist)
- **ASVspoof 2019 LA**: English test set, used only to check the test code, through [a copy on Hugging Face](https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA). [Project site](https://www.asvspoof.org/)

## Responsible use

This project only detects cloned voices. It uses existing research datasets and does not generate clones of anyone.
