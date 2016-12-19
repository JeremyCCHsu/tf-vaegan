# VAE-GAN in Tensorflow
Implementation (with modifications) of [*Autoencoding beyond pixels using a learned similarity metric*](https://arxiv.org/abs/1512.09300)  

As the name indicates, VAE-GAN replaces GAN's generator with a variational auto-encoder, resulting in a model with both inference and generation components.

*Please refer to their official Github for details*: https://github.com/andersbll/autoencoding_beyond_pixels


## Dependencies
1. Tensorflow  
2. Matplotlib  
3. PIL

## Examples

### Interpolating Chinese Characters



## Discussion
The gamma parameter in Eq. (9) is a trade-off between *style* and *content* as mentioned in the paper. In my experiment, if gamma is set too small (such as 1e-5), the content could be lost, thus unable to reconstruct the input. However, random sampls directly generated from the latent space could be realistic in this case. Setting gamma to a larger value (say, 0.1), we ended up with a good reconstruction, but the random samples could be less reasonable to the eyes.