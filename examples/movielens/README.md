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
python movielens_tpp.py N M
```

VI with grouped latents:
```sh
python movielens_grouped.py N M
```

RWS:
```sh
python movielens_discrete_variance_rws.py N M
```

VI with all latents grouped:
```sh
python movielens_discrete_Global_K_variance_rws.py N M
```

With $N \in \lbrace 30,200 \rbrace$, $M \in \lbrace 10,50,100 \rbrace$
