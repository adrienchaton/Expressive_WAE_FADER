# Assisted Sound Sample Generation with Musical Conditioning in Adversarial Auto-Encoders

This repository is a companion to the DAFx19 submission:
Adrien Bitton, Philippe Esling, Antoine Caillon, Martin Fouilleul, "Assisted Sound Sample Generation with Musical Conditioning in Adversarial Auto-Encoders".

Preliminary works can be accessed at https://github.com/nebularnoise/serge.
An Arxiv pre-print is also being processed and online soon.


━━━ ☆ﾟ.*･｡ﾟ This demonstration repository is under construction, sound examples, visualizations and additional results from the final experiment will be uploaded, as well as codes and plugin implementation after the reviewing process.


## ✿ ABSTRACT

*Deep generative neural networks* have thrived in the field of computer vision, enabling unprecedented intelligent image processes. Yet the results in audio remain less advanced and many applications are still to be investigated. Our project targets real-time sound synthesis from a reduced set of high-level parameters, including semantic controls that can be adapted to different sound libraries and specific tags. These generative variables should allow expressive modulations of target musical qualities and continuously mix into new styles.

To this extent we train *auto-encoders* on an orchestral database of individual note samples, along with their intrinsic attributes: note class, *timbre domain* (an instrument subset) and *extended playing techniques*. We condition the decoder for explicit control over the rendered note attributes and use *latent adversarial training* for learning expressive style parameters that can ultimately be mixed. We evaluate both generative performances and correlations of the attributes with the latent representation. Our ablation study demonstrates the effectiveness of the *musical conditioning* mechanisms. 

The proposed model generates notes as magnitude spectrograms from any probabilistic latent code samples, with expressive control of orchestral timbres and playing styles. Its training data subsets can directly be visualized in the 3-dimensional latent representation. Waveform rendering can be done offline with the *Griffin-Lim algorithm* (GLA). In order to allow real-time interactions, we fine-tune the decoder with a pretrained magnitude spectrogram inversion network and embed the full waveform generation pipeline in a *plugin*. Moreover the encoder could be used to process new input samples, after manipulating their latent attribute representation, the decoder can generate sample variations as an *audio effect* would. Our solution remains rather light-weight and fast to train, it can directly be applied to other sound domains, including an *user's libraries* with *custom sound tags* that could be mapped to specific generative controls. As a result, it fosters creativity and intuitive audio style experimentations.

♪ **WAE-Fader implementation:** the model uses a non-conditional encoder with *Batch-Norm* (BN), the decoder is *conditional* with *Adaptive Instance Normalization* (AdaIN) and *Feature Wise Linear Modulation* (FiLM). The WAE latent regularization uses *MMD-Rbf* and isotropic *gaussian prior*. The encoder adversarially trains against a *Fader* latent discriminator to enforce the decoder conditioning. Waveform synthesis is made realtime capable by pairing the decoder with a *Multi-head Convolution Neural Network* (MCNN).

## ✿ Additional visualizations

♪ **Test spectrogram reconstructions**

We plot log-scaled input Mel-spectrograms and corresponding auto-encoder reconstructions. We only display test set notes to assess the quality and generalization power of the models.

♪ **Latent spaces and adversarial training**

We plot the learned latent representations of different models and data subsets. 3-dimensional scatter plots are the raw latent coordinates, 2-dimensional projections are done with t-SNE (t-distributed Stochastic Neighbor Embedding). We use colors to highlight the different attribute subsets: semitones, octaves and styles (playing techniques or timbre domains).

For WAE-Fader models, we as well show the evolution of the latent representation throughout the model training with adversarial latent classification. In this setting, the Fader latent discriminator tries to classify style attributes from the non-conditional encoder output **z**. In turn and after the **α-warmup** has started, the encoder tries to fool the discriminator so that its latent representation cannot be properly classified. It encourages an attribute-free latent code and pushes the decoder to learn its conditioning.

At first is pre-classification, only the Fader classification is optimized without adversarial back-propagation in the encoder (α=0). After this and until the end of the first half of training epochs, (α:0➔4) is gradually increased to its target value and then remains fixed until the end of the training.

<img src="https://github.com/adrienchaton/Expressive_WAE_FADER/blob/master/figures/RAW3D_anim.gif" width="183" height="646">  <img src="https://github.com/adrienchaton/Expressive_WAE_FADER/blob/master/figures/TSNE2D_anim.gif" width="378" height="574">

## ✿ Sound examples

We may also upload on soundcloud as github audio streaming is often slow.

♪ **Test set reconstructions inverted with GLA or MCNN**

We give test set spectrogram reconstructions inverted to waveform with either GLA (iterative) or MCNN (feed-forward and realtime capable) to allow for individual listening and evaluation of the current audio qualities we achieved.

♪ **Random conditional note generations with WAE-style or WAE-Fader**

We give some random note samples, that were generated in the same way as for the note-conditional generative evaluation. Given a random latent point sampled from the model prior and random note targets with octave either in {3-4} (common to all instrument tessitura) or in {0-8} (full orchestral range), we decode on each model's style attribute and invert to waveform with either GLA or MCNN. Here we control the categorical style classes that correspond to each training data subset.

We can see that accordingly to the evaluations of WAE-style, it does not render meaningful audio variations with respect to the target style attributes (both for playing techniques and timbres). Whereas, WAE-Fader produces consistent sound variations that express the desired target qualities. It demonstrates its ability to efficiently learn and generalize the conditional controls to random samples from the prior, as validated by our final evaluations.

♪ **Expressive style and timbre synthesis with WAE-Fader**

Since we demonstrated the ability of WAE-Fader to generate and control random note samples consistently with each of its attribute subsets, we further explore the potential expressivity enabled when continuously mixing the learned *fader* style variables. We could for instance consider some Pizzimolo as *0.5 Pizzicato-secco + 0.5 Tremolo* or Clariolin as *0.5 Clarinet + 0.5 Violin* or .. ?? ☆ﾟ.*･｡ﾟ

## ✿ Upcoming

+ Final plugin development and demonstration
+ Code release and experiments on other sound libraries
+ Improved audio-quality and end-to-end waveform modelling
