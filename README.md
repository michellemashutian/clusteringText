# ClusteringText

A repository for exploring clustering techniques in natural language processing (NLP), with a focus on analyzing and visualizing textual datasets. This project demonstrates the implementation of unsupervised learning methods to group similar text documents effectively.

## Features

- Preprocessing pipeline for text datasets (can preprocess data in Chinese)

- Multiple clustering algorithms (K-Means, MinibatchKmeans, Birch, AffinityPropagation, AgglomerativeClustering, DBSCAN)

- Support for various vectorization methods (VSM, LSI, LDA)

- Easy integration with custom datasets (given text and keywords)

## Installation

+ Clone the repository:
```
git clone https://github.com/michellemashutian/clusteringText.git
cd clusteringText
```

+ Install required dependencies:
```
pip install -r requirements.txt
```
## Usage

Prepare your dataset in txt format with columns containing text data.

Run the main script:
```
python main.py
```


## Dependencies
```
gensim==4.3.3
jieba==0.42.1
numpy==2.2.1
scikit_learn==1.6.0
```


## Contact

For any questions or feedback, feel free to reach out via issues or email me at mashutian0608@hotmail.com

## Happy Clustering!

