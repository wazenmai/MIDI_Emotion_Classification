# MIDI_Emotion_Classification


Temporary repository for midi classifiers.
- input: midi path
- option : HV, Arousal, Valence
    - output HV: The output score of Q1,Q2,Q3,Q4
    - output Arousal: The output score of High Arousal, Low Arousal
    - output Valence: The output score of High Valence, Low Valence

Refer to notebook/DataSplit and ML(Logistic Regression) for ml classifier and notebook/Midi Inference for dl classifier.

### Current Model
[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130) (as SAN)

```
conda create -n YOUR_ENV_NAME python=3.8
conda activate YOUR_ENV_NAME
pip install -r requirements.txt
```
### Training
```
cd midi_dl
python build_midi_vocab.py

# if AV case
python train_test.py --model Net --pipeline arousal_valence

# if 2class case (Arousal or Valence)
python train_test.py --model Net_2Class --pipeline arousal --runner gpu0_lre4_wde6
python train_test.py --model Net_2Class --pipeline valence --runner gpu0_lre4_wde6
```

### Inference
Arousal_Valence [Q1,Q2,Q3,Q4]
```
python inference.py --midi_path your_midi --types AV
```
```
========
./dataset/sample_data/example_generative.mid  is emotion Q1
Inference values:  tensor([[ 1.4644,  1.1517, -1.4332, -1.2267]], device='cuda:0',
       grad_fn=<AddmmBackward>)
```
Arousal
```
python inference.py --midi_path your_midi --types Arousal
```
```
========
./dataset/sample_data/example_generative.mid  is emotion HA
Inference values:  tensor([[ 1.0950, -1.0352]], device='cuda:0', grad_fn=<AddmmBackward>)
```
Valence
```
python inference.py --midi_path your_midi --types Valence
```
```
========
./dataset/sample_data/example_generative.mid  is emotion LV
Inference values:  tensor([[-0.0115,  0.1326]], device='cuda:0', grad_fn=<AddmmBackward>)
```