export num_steps=40000
export num_trials=10
export key=314

## Epsilon Greedy

python -W ignore run_mnist_bandit.py run-epsilon-greedy --base_path output --num_trials $num_trials --agent FLoRES --num_steps $num_steps --key $key

# python -W ignore run_mnist_bandit.py run-epsilon-greedy --base_path output --num_trials $num_trials --agent LoFi --num_steps $num_steps --key $key

# python -W ignore run_mnist_bandit.py run-epsilon-greedy --base_path output --num_trials $num_trials --agent LRKF --num_steps $num_steps --key $key

# python -W ignore run_mnist_bandit.py run-epsilon-greedy --base_path output --num_trials $num_trials --agent adamw --num_steps $num_steps --key $key

# python -W ignore run_mnist_bandit.py run-epsilon-greedy --base_path output --num_trials $num_trials --agent muon --num_steps $num_steps --key $key

## Thompson Sampling

python -W ignore run_mnist_bandit.py run-ts --base_path output --num_trials $num_trials --agent FLoRES --num_steps $num_steps --key $key

# python -W ignore run_mnist_bandit.py run-ts --base_path output --num_trials $num_trials --agent LoFi --num_steps $num_steps --key $key

# python -W ignore run_mnist_bandit.py run-ts --base_path output --num_trials $num_trials --agent LRKF --num_steps $num_steps --key $key

# python -W ignore run_mnist_bandit.py run-ts --base_path output --num_trials $num_trials --agent adamw --num_steps $num_steps --key $key

# python -W ignore run_mnist_bandit.py run-ts --base_path output --num_trials $num_trials --agent muon --num_steps $num_steps  --key $key