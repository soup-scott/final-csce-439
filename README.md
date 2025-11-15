### CSCE 429 Team 10 Defense Repository

## Pre-training
If you want to train the model, you must first prepare samples in a .joblib format. Follow the steps below to do so.

```bash
python3 -m featuregen
python3 send_samples.py DIRECTORY --label LABEL --recursive
```

Now your samples are in .jsonl format. Next transform into .joblib for reuse in training.

```bash
python3 train/save_features.py
```

## Training
Now to train the model.

```bash
cd train
python3 simpleRF_train.py
```

## Testing
Now to test the model.

```bash
python3 benchmark.py
```

## Deployment
To deploy the model to the webserver.

```bash
python3 -m defender
```