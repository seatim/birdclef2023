
<p align="center">
    <img alt="Spectogram image" src="https://raw.githubusercontent.com/seatim/birdclef2023/main/tests/data/train_images/helgui/XC503001.ogg-2-5.png"/>
</p>

# birdclef2023

Python package and programs for developing classification models for the
[BirdCLEF 2023](https://www.kaggle.com/competitions/birdclef-2023) coding
competition.

[![CI](
https://github.com/seatim/birdclef2023/actions/workflows/adak-ci.yml/badge.svg)](
https://github.com/seatim/birdclef2023/actions/workflows/adak-ci.yml)

## 🛠 Setup

### Code

Python 3.7+ is required.

Create a virtual environment:

    virtualenv VENV_DIR

Activate virtualenv:

    . VENV_DIR/bin/activate

Install dependencies:

    pip install -r requirements.txt

### Data

Join the competition to get access to the data, then unpack it into the
`data/birdclef-2023` directory.

This code supports training on the data used in prior years' competitions, i.e.
[BirdCLEF 2021](https://www.kaggle.com/competitions/birdclef-2021) and
[BirdCLEF 2022](https://www.kaggle.com/competitions/birdclef-2022).

If using these datasets, unpack them to the `data/birdclef-2021` and
`data/birdclef-2022` directories.

## 📚 Usage

### Use without installing

The software in this repo can be used directly without being installed.  You
can either run the programs from this directory or point `PYTHONPATH` to this
directory and run the programs from anywhere.

### Make images

Use `make_images_from_audio.py` to generate training and validation images
from audio files.  By default the entire collection of audio files is used.
To make smaller training and validation image sets for testing, use the
`--max-images-per-file` (`-m`) and `--max-paths-per-class` (`-M`) options.

Example:

    python make_images_from_audio.py -m 5 -M 5 -i data/train_images.m5.M5/

### Train a model

Use `train_classifier.py` to create an image classification model.

For example, to train a classifier on the small training set created in the
previous example, run this command:

    python train_classifier.py -i data/train_images.m5.M5/ -B '' -D ''

### Combined directory

The training program will copy images from the input training sets into a
"combined" directory, which is located at `data/train_images.combined` by
default.  The model produced by the training will be saved to this directory
also.

### Evaluate a model

Use `evaluate.py` to evaluate a classifier.  `evaluate.py` reads file names
from standard input, uses a given model to classify each image, and outputs
performance metrics.

For example, to evaluate a classifier on the validation set created earlier,
run this bash command:

    find val_images.m5.M5/ -type f | python evaluate.py PATH_TO_MODEL

The predictions made by the classifier can be saved to a file for later
analysis using the `--save-preds` (`-s`) option.  For example:

    find val_images.m5.M5/ -type f | python evaluate.py PATH_TO_MODEL -s preds.csv

### Analyze predictions

Use `analyze_preds.py` to gain additional insights about classifier
performance.

For example, to show some statistics of the predictions, use the `--show-stats`
(`-s`) option:

    python analyze_preds.py -s PREDS_FILE

To see a full list of analysis options, run this command:

    python analyze_preds.py --help

## 🚦 Testing

Install test dependencies:

    pip install -r tests/requirements.txt

To run tests:

    python -m unittest

## License

[XC503001](https://xeno-canto.org/503001) by [Dawie de Swardt](
https://xeno-canto.org/contributor/JDCTKFMBKV)
and [XC390820](https://xeno-canto.org/390820) by [Tieme](
https://xeno-canto.org/contributor/OUZGUMHSWU) have been transcoded to [Ogg](
https://en.wikipedia.org/wiki/Ogg) format and are redistributed under the
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

All other files are released under the MIT License.  See LICENSE file.
