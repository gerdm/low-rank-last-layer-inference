export num_steps=40000
export num_trials=10
export key=314

# Laplace
python -W ignore run_mnist_bandit.py run-predictive-bayes --base_path output --num_trials $num_trials --agent adamw --num_steps $num_steps --key $key
python -W ignore run_mnist_bandit.py run-ts --base_path output --num_trials $num_trials --agent adamw --num_steps $num_steps --key $key

## Lofi
python -W ignore run_mnist_bandit.py run-predictive-bayes --base_path output --num_trials $num_trials --agent LoFi --num_steps $num_steps --key $key
python -W ignore run_mnist_bandit.py run-ts --base_path output --num_trials $num_trials --agent LoFi --num_steps $num_steps --key $key

## HiLoFi
python -W ignore run_mnist_bandit.py run-predictive-bayes --base_path output --num_trials $num_trials --agent FLoRES --num_steps $num_steps --key $key
