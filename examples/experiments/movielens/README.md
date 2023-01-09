MovieLens
=========

<!-- put a description of movielens here -->

Data
====

```sh
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip -d ml-100k
python make_data.py
```

Running scripts
===============

VI:
```sh
python runner.py dataset=movielens model=movielens training.inference_method=elbo training.N=N training.M=M
```

Locally importance weighted VI:
```sh
python runner.py local=True dataset=movielens model=movielens training.inference_method=elbo training.N=N training.M=M
```

RWS:
```sh
python runner.py dataset=movielens model=movielens_discrete training.inference_method=rws training.N=N training.M=M
```


With $N \in \lbrace 30,200 \rbrace$, $M \in \lbrace 10,50,100 \rbrace$
