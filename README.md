# META-RS

This is the companion code for the paper "Meta-Learning the Search Distribution of Black-Box Random 
Search Based Adversarial Attacks"  by Yatsura et al. published in NeurIPS 2021.
The code allows the users to reproduce and extend the results reported in the study.
Please cite the above paper when reporting, reproducing or extending the results.

## Purpose of the project

This software is a research prototype, solely developed for and published as part of the publication. It will neither be
maintained nor monitored in any way.

## Usage example

The script `meta_learned_square_attack/main.py` can be used for both meta-training and evaluation. When no training is required i. e. we 
run the attack in evaluation mode, use command line flag `--test`

Examples:

Meta-training the controllers:

```python meta_learned_square_attack/main.py --model resnet18 --n_images 1000 --n_iter 1000 --n_epochs 10 --bs 100 --meta_lr 0.03 --color mlp --step_size mlp --meta_schedule cosine --p 0.8 --loss ce --update_threshold 0 --relaxed_squares --momentum 0.99 --temperature 1 --seed 0 --n_hidden 2 --hidden_size 10```

Evaluation on CIFAR10 robustbench models:

```python meta_learned_square_attack/main.py --model Ding2020MMA --n_images 1000 --n_iter 5000 --n_epochs 1 --bs -1 --color controllers/color_controller.pkl --step_size controllers/step_size_controller.pkl --meta_schedule cosine --loss margin --test --seed 0```

Evaluation on ImageNet robustness models:

```python meta_learned_square_attack/main_imagenet.py --datapath <PATH_TO_IMAGENET> --model resnet18 --model_dir <PATH_TO_A_PRETRAINED_MODEL> --n_images 1000 --n_iter 5000 --n_epochs 1 --bs 1000 --color controllers/color_controller.pkl --step_size controllers/step_size_controller.pkl --loss margin --test --seed 0```

## License

META-RS is open-sourced under the AGPL-3.0 license. See the [LICENSE](LICENSE) file for details.
