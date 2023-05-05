MovieLens
=========

Data
====

```sh
cd movielens
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
mkdir data/
mkdir results/
python make_data.py
```

Running scripts
===============
```sh
python vi.py
python movielens_exp.py
```

Bus Breakdown
=============

Data
====

```sh
cd bus_breakdown
wget -O Bus_Breakdown_and_Delays.csv https://data.cityofnewyork.us/api/views/ez4e-fazm/rows.csv
mkdir data/
mkdir results/
python make_data.py
```

Running scripts
===============
```sh
python vi.py
python bus_breakdown_exp.py
```
