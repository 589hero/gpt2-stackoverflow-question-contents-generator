# gpt2-stackoverflow-question-contents-generator


* 질문의 제목을 입력하면 이를 바탕으로 stackoverflow 스타일의 질문 내용을 생성해줍니다.
* Fine-tuning data & code
  * <a href="https://www.kaggle.com/imoore/60k-stack-overflow-questions-with-quality-rate">Data</a> 
  * Use only Y="HQ(High Quality)" data and a length of sentences less than 1000.
  * <a href="https://drive.google.com/file/d/12NKzoniQ9qS9roRSOOcQaqIYoyyJmOTO/view?usp=sharing">Code</a> 

* Model
  * GPT2-medium


## How to build & run
1. docker build -t "Your own docker image name" .
2. docker run -it -p 8051:8051 "Your own docker image name":latest

## Reference
- https://github.com/dleunji/resume
- https://github.com/ainize-team/tabtab
