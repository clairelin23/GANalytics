# StyleGAN-PyTorch

This is a simple but complete pytorch-version implementation of Nvidia's Style-based GAN[3]. We've train this model on [the oxford flowers dataset](https://www.tensorflow.org/datasets/catalog/oxford_flowers102) and a subset of the Pokemon dataset, you can download our pre-trained model to evaluate or continue training by yourself.


### Overview

### With and without noise

### Style-mixing

### Pre-trained model

*Not available yet.*

### Save model & continue training

When you train the model yourself, parameters of the model will be saved to `./checkpoint/trained.pth` every 1000 iterations. You can set `is_continue` to `True` to continue training from your pre-trained model.

## Performance

### Loss curve

*Not available yet.*

### [Fréchet Inception Distance](https://arxiv.org/abs/1706.08500)

We use Fréchet Inception Distance to estimate the performance of our implementation. We use an edited version (changes to it will not affect the score it gives) of [mseitzer's work](https://github.com/mseitzer/pytorch-fid) to estimate our model's performance.

*Not available yet.*
 
## References

[1] Mescheder, L., Geiger, A., & Nowozin, S. (2018). Which Training Methods for GANs do actually Converge? Retrieved from http://arxiv.org/abs/1801.04406

[2] Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017). Progressive Growing of GANs for Improved Quality, Stability, and Variation. 1–26. Retrieved from http://arxiv.org/abs/1710.10196

[3] Karras, T., Laine, S., & Aila, T. (2018). A Style-Based Generator Architecture for Generative Adversarial Networks. Retrieved from http://arxiv.org/abs/1812.04948