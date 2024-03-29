[![Docker Image CI](https://github.com/wavescholar/wavelang/actions/workflows/docker-image.yml/badge.svg)](https://github.com/wavescholar/wavelang/actions/workflows/docker-image.yml)

# wavelang

This repo is for testing out some large language models on sentiment data. 

We test OpenAI, Google Gemini, a BERT models. We're comparing the ability to 
classify financial sentiment data into three ordinal classes. The data is labeled by independent experts and
is partitioned into four levels of agreement. 



### Dockerbuild 

```
docker build -t wavelang -f ./Dockerfile .
docker tag wavelang wavescholar/wavelang:latest
docker push -a wavescholar/wavelang
```
