# gpt2-stackoverflow-question-contents-generator

* Fine-tuning data
  * <a href="https://www.kaggle.com/imoore/60k-stack-overflow-questions-with-quality-rate">Kaggle</a> 
  * Use only Y="HQ(High Quality)" data and a length of sentences less than 1000.

* Model
  * GPT2-medium


# How to build
1. docker build -t "Your own docker image name" .
2. docker run -it -p 8051:8051 "Your own docker image name":latest
