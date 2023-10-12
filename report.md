# EvaDB Project 1 Report

**Name: Chi-Ting Liu, GTID: 903925209**

[Github Link](https://github.com/Gting6/EvaDB_1), [Google Colab Link](https://colab.research.google.com/drive/1oVVlceoH1MiLBeESKhxiqs3m0yW7p6bx?usp=sharing)

## Topic: Integrate EvaDB with AWS Rekognition services

[Amazon Rekognition](https://aws.amazon.com/rekognition/) stands as a cloud-based computer vision platform leveraging deep learning to interpret images, offering a diverse spectrum of image and video analysis functionalities. [EvaDB](https://evadb.readthedocs.io/en/stable/index.html) enables software developers to build AI apps in a few lines of code. Its powerful SQL API simplifies AI app development for both structured and unstructured data. In this project, we integrate Amazon's Rekognition Service with EvaDB. **One can compare faces, detect faces, detect labels, detect texts with a simple SQL instruction.**

### Implementation details

Follow the instructions on [Writing a custom function](https://evadb.readthedocs.io/en/stable/source/reference/ai/custom.html#part-1-writing-a-custom-function) in EvaDB. Modify the input / output signature and process the result to fit the format of AWS Rekognition Services.

### Sample Input / Output

We first load the images into `faceDemo`. There are photos from 3 people: liu, chen and wen.

- `liu1.jpg`, `liu2.jpg`, `liu3.jpg` contains only liu's photo
- `chen.jpg` contains only chen's photo
- `liuwen.jpg` have both liu and wen inside

**We use liuwen.jpg as target. We successfully retrieve all images with liu or wen from a simple SQL query.**

```
SELECT name, AWSCompareFaces(data, Open('./liuwen.jpg'))
FROM faceDemo
WHERE AWSCompareFaces(data, Open('./liuwen.jpg')) > [80.0]
ORDER BY AWSCompareFaces(data, Open('./liuwen.jpg'));
```

[![Screenshot 2023-10-12 at 10 56 36 AM](https://user-images.githubusercontent.com/46078333/274640986-702bef3a-4b92-4218-a3fd-b3e233fa3475.png)](https://user-images.githubusercontent.com/46078333/274640986-702bef3a-4b92-4218-a3fd-b3e233fa3475.png)

There are 4 more examples on [Github](https://github.com/Gting6/EvaDB_1), [Google Colab](https://colab.research.google.com/drive/1oVVlceoH1MiLBeESKhxiqs3m0yW7p6bx?usp=sharing). In conclusion, we can easily perform face comparison, face detection, label detection and text detection with a simple SQL instruction.

### Metrics

**Time**

We measure the execution time with Python's `time.time()` module

For the dataset we use, the result is as following:

- Compare 1 image with 6 images (awsCompareFaces): 19 ms
- Detect faces on 6 images (awsDetectFaces): 4 ms
- Detect labels on 1 image (awsDetectLabels): 5.8 ms
- Detect text on 1 image (awsDetectText): 7.5 ms

As a reference comparison, we also measure the time for [Similarity function](https://colab.research.google.com/github/georgia-tech-db/eva/blob/staging/tutorials/11-similarity-search-for-motif-mining.ipynb#scrollTo=f54cfe6b) in evaDB in same setting :

To be more specific, we measure the following instructions

```
 SELECT name,Similarity(
  SiftFeatureExtractor(Open('liu1.jpg')),
  SiftFeatureExtractor(data)
) FROM faceDemo
```

Result = <img src="/Users/gtingliu/Library/Application Support/typora-user-images/Screenshot 2023-10-12 at 12.31.09 PM.png" alt="Screenshot 2023-10-12 at 12.31.09 PM" style="zoom:50%;" />, Exexution time = 575 ms

As for our implementation, the instructions are 

```
SELECT name, AWSCompareFaces(data, Open('./liu1.jpg'))
FROM faceDemo
```

Result = <img src="/Users/gtingliu/Library/Application Support/typora-user-images/Screenshot 2023-10-12 at 12.32.16 PM.png" alt="Screenshot 2023-10-12 at 12.32.16 PM" style="zoom:50%;" />, Execution time = 10 ms

Conclusion: 

- If your application requires face comparison, such as facial recognition for face search, **our implementation can deliver the desired results in just 1/50th of the time.**

- However, please be aware that this comparison is somewhat simplistic due to the differences in output. A more rigorous benchmark is necessary for thorough analysis. Nonetheless, it does indicate that AWS Rekognition can generate the desired output quickly.

**Budget**

- The pricing of AWS rekognition is approximately $0.0010 per image.
- Details can be found on [Pricing](https://aws.amazon.com/rekognition/pricing/?nc=sn&loc=4)

### Lessons Learned

- EvaDB is a highly versatile service capable of seamlessly integrating a wide range of AI functionalities.
- AWS Rekognition is a user-friendly service adept at executing intricate computer vision tasks with ease.
- Combining these two results in a powerful AI-powered database.

### Challenges

- Input/Output Format: The AWS Rekognition service accepts input as bytes, while EvaDB's forward function requires input in the form of a numpy array. I spent some time in understanding these two formats and implementing the necessary conversions.
- There is an existing issue with the compare_faces function in the AWS Rekognition service, as detailed in this [link](https://repost.aws/questions/QUWiAxrjlMS_qTyk8o7WtDyQ/old-bug-in-comparefaces-jpg-throws-error-if-no-face-when-will-it-be-fixed). If there is no face detected in the image, calling the compare_faces function will lead to an error. It took me some time to identify this issue.

### References

1. [AWS Rekognition Documents](https://aws.amazon.com/rekognition/)
2. [EvaDB Document](https://evadb.readthedocs.io/en/stable/index.html)