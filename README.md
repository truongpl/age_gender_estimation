This project aims to use Tensorflow 2.0 to predict: gender - age - ethics of a face image

# UTK Dataset
UTK dataset is [here](https://susanqq.github.io/UTKFace/). The dataset has 20k+ images, each image has [age,gender,race] annotated with following format:
age_gender_race_imagename.

# Source structure

| Location             |  Content                                   |
|----------------------|--------------------------------------------|
| `/data`              | Unzip utk dataset to this folder           |
| `face_model.py        `   | Face multi task using pretrained inception v3        |
| `predict.py ` | Predict sample imgs |
| `predict.py ` | create_dataset.py |
| `params.json ` | Settings |

# Detail process
UTK is face in the wild, at first MTCNN face detector is applied for each image in dataset to reduce the effect of scene on model. Newly created dataset is produced after create_dataset.py

```
python create_dataset.py
```

Three heads (age, gender, races) are placed on top of InceptionV3. To use pretrained model, the backbone need to be frozen for some epochs and unfrozen for long run:

```
python face_model.py
```

After 100 epochs, accuracy in val set is [92% for gender, 80% for race, 42% for age]. The model is probably overfit on age head.