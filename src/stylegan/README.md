# StyleGAN-PyTorch

This is a pytorch-version implementation of Nvidia's Style-based GAN. I have trained this model on [the oxford flowers dataset](https://www.tensorflow.org/datasets/catalog/oxford_flowers102) and a subset of the Pokemon dataset.

## Training Setup
1. Create a directory name `dataset` under the same directory as `train_stylegan_model.py`.
2. Download [the oxford flowers dataset](https://www.tensorflow.org/datasets/catalog/oxford_flowers102) and put it in the `dataset` directory.
3. You can modify `config.py` to add or edit dataset configurations, such as dataset location, and dataset name.
4. To start training the model, run `python train_stylegan_model.py {dataset name}`. 
You can modify training parameters at the top of file as needed.


## Model Checkpoints
When the model is trained, parameters checkpoint will be saved to `./checkpoint/trained.pth` every 1000 iterations.

## Evaluation Metric
Fréchet Inception Distance is used to estimate the performance. Please see `calculate_scores.py` for more details.

## Result Visualization
Please use `plot_results.ipynb` for visualizing training results

## Next steps
Implementing style-mixing feature is considered as a next step.