# Project Topic: Integrate EvaDB with AWS Rekognition services
![Screenshot 2023-10-12 at 11 10 47 AM](https://github.com/Gting6/EvaDB_1/assets/46078333/9c56a682-e053-47cf-9345-dc27483969f6)


## Introduction
[Amazon Rekognition](https://aws.amazon.com/rekognition/) stands as a cloud-based computer vision platform leveraging deep learning to interpret images, offering a diverse spectrum of image and video analysis functionalities. [EvaDB](https://evadb.readthedocs.io/en/stable/index.html) enables software developers to build AI apps in a few lines of code. Its powerful SQL API simplifies AI app development for both structured and unstructured data. In this project, we integrate Amazon's Rekognition Service with EvaDB. **One can compare faces, detect faces, detect labels, detect texts with a simple SQL instruction.**

## Run the code
- Google Colab (Recommended)
  - [Run on Google Colab](https://colab.research.google.com/drive/1oVVlceoH1MiLBeESKhxiqs3m0yW7p6bx?usp=sharing)
- Local Machine
  - Set your aws credential and region to `~/.aws/credential` and `~/.aws/config`
  - `pip install -r requirements.txt`
  - Put all files in `image` into `./`
  - Put all files `aws*` in `evadb/functions/`
  - Put `run_evadb.py` in `./`
  - `python -m run_evadb`

## Result
We first load the images into `faceDemo`. There are photos from 3 people: liu, chen and wen.
- `liu1.jpg`, `liu2.jpg`, `liu3.jpg` contains only liu's photo
- `chen.jpg` contains only chen's photo
- `liuwen.jpg` have both liu and wen inside

We also load `text.jpg` to `textImg` and `dog.jpg` to `objectImg` to demonstrate text recognition and object detection function.

### Demo
**1. We use liu1.jpg as target. We successfully retrieve all images with liu from a simple SQL query.**
```
SELECT name, AWSCompareFaces(data, Open('./liu1.jpg'))
FROM faceDemo
WHERE AWSCompareFaces(data, Open('./liu1.jpg')) > [80.0];
```
![Screenshot 2023-10-12 at 10 55 08 AM](https://github.com/Gting6/EvaDB_1/assets/46078333/f135cd70-0608-4f65-9c24-ca7dc1f9df35)

**2. We use liuwen.jpg as target. We successfully retrieve all images with liu or wen from a simple SQL query.**
```
SELECT name, AWSCompareFaces(data, Open('./liuwen.jpg'))
FROM faceDemo
WHERE AWSCompareFaces(data, Open('./liuwen.jpg')) > [80.0]
ORDER BY AWSCompareFaces(data, Open('./liuwen.jpg'));
```
![Screenshot 2023-10-12 at 10 56 36 AM](https://github.com/Gting6/EvaDB_1/assets/46078333/702bef3a-4b92-4218-a3fd-b3e233fa3475)

**3. We analysis faces such as age, gender, emotion, glasses, ... etc in a simple SQL query**
```
SELECT name, AWSDetectFaces(data)
FROM faceDemo
```
![Screenshot 2023-10-12 at 10 58 19 AM](https://github.com/Gting6/EvaDB_1/assets/46078333/f600c261-22c1-405b-af2b-715c415c2cef)

**4. We detect objects in a simple SQL query**
```
SELECT name, AWSDetectLabels(data)
FROM objectImgs
```
![Screenshot 2023-10-12 at 11 01 08 AM](https://github.com/Gting6/EvaDB_1/assets/46078333/c4bb3469-aa10-4f16-a62a-5324dcd0a334)

**5. We detect texts in a simple SQL query**
```
SELECT name, AWSDetectTexts(data)
FROM textImgs
```
![Screenshot 2023-10-12 at 11 01 49 AM](https://github.com/Gting6/EvaDB_1/assets/46078333/c0e01e13-d7d4-4553-93b9-964f28c3e01a)

