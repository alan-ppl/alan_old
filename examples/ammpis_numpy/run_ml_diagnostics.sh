python ml_diagnostic.py -N 1 -K 1 -T 100
python ml_diagnostic.py -N 1 -K 1 -T 100 -p
python ml_diagnostic.py -N 1 -K 1 -T 5000
python ml_diagnostic.py -N 1 -K 1 -T 5000 -p
echo "Done with N=1, K=1"

python ml_diagnostic.py -N 1 -K 5 -T 100
python ml_diagnostic.py -N 1 -K 5 -T 100 -p
python ml_diagnostic.py -N 1 -K 5 -T 5000
python ml_diagnostic.py -N 1 -K 5 -T 5000 -p
echo "Done with N=1, K=5"

python ml_diagnostic.py -N 5 -K 1 -T 100
python ml_diagnostic.py -N 5 -K 1 -T 100 -p
python ml_diagnostic.py -N 5 -K 1 -T 5000
python ml_diagnostic.py -N 5 -K 1 -T 5000 -p
echo "Done with N=5, K=1"

python ml_diagnostic.py -N 5 -K 5 -T 100
python ml_diagnostic.py -N 5 -K 5 -T 100 -p
python ml_diagnostic.py -N 5 -K 5 -T 5000
python ml_diagnostic.py -N 5 -K 5 -T 5000 -p
echo "Done with N=5, K=5"

python ml_diagnostic.py -N 50 -K 10 -T 1000
python ml_diagnostic.py -N 50 -K 10 -T 1000 -p
echo "Done with N=50, K=10"
