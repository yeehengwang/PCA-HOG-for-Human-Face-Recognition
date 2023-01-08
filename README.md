# README

---

# ***PCA & HOG for Human Face Recognition***

  ******

> ***Face recognition is a very typical task for machine learning research. In this project, we use two methods, Histograms of Oriented Gradients and Principal Component Analysis, to process the initial face images. SVM is used as a baseline classifier throughout the study.***
> 

---

## Downloading Datasets

        We use the data set named “The Labeled Faces in the Wild” with 13233 samples and 5749 classes. And each of the face image is 62*47 with 2914 pixels. “The Labeled Faces in the Wild” is a public benchmark for face verification, also known as pair matching.

You can download the datasets from: [http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/)

---

## Running The Program

### R****equirements****

time

matplotlib

sklearn

os

### Getting Started

```bash
cd ML_FINAL
#running HOG.py
python3 HOG.py
#running PCA.py
python3 PCA.py

```

---

## Results Preview

### HOG

![Figure_1.png](README%206b242b0f342941d49428118d50a4ab93/Figure_1.png)

![Figure_2.png](README%206b242b0f342941d49428118d50a4ab93/Figure_2.png)

### PCA

![Figure_3.png](README%206b242b0f342941d49428118d50a4ab93/Figure_3.png)

![Figure_4.png](README%206b242b0f342941d49428118d50a4ab93/Figure_4.png)

---

## Reference

**[1]:** [N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," *2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05)*, 2005, pp. 886-893 vol. 1, doi: 10.1109/CVPR.2005.177.](https://ieeexplore.ieee.org/document/1467360)

**[2]: [**urk, Matthew, & Alex Pentland. “Eigenfaces for Recognition”. *Journal of Cognitive Neuroscience* 3, 1 (1991.1.1): 71–86. https://doi.org/10.1162/jocn.1991.3.1.71.](https://www.cin.ufpe.br/~rps/Artigos/Face%20Recognition%20Using%20Eigenfaces.pdf)