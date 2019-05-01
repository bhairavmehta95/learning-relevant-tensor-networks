%sh 

echo "Running BatchSize 512"
python -m experiments.driver_ucg --filename=bsz512

echo "Running BatchSize 1024"
python -m experiments.driver_ucg --batch-size=1024 --filename=bsz1024

echo "Running BatchSize 2048"
python -m experiments.driver_ucg --batch-size=2048 --filename=bsz2048

echo "Running BatchSize 512"
python -m experiments.driver_ucg --batch-size=4096 --filename=bsz4096

echo "Running BatchSize 512"
python -m experiments.driver_ucg --batch-size=8192 --filename=bsz8192
