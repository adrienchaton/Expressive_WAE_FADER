# Assisted Sound Sample Generation with Musical Conditioning in Adversarial Auto-Encoders

This repository is a companion to the DAFx19 submission:
Adrien Bitton, Philippe Esling, Antoine Caillon, Martin Fouilleul, "Assisted Sound Sample Generation with Musical Conditioning in Adversarial Auto-Encoders".

Preliminary works can be accessed at https://github.com/nebularnoise/serge.
An Arxiv pre-print is also being processed and online soon.


✿✿✿✿✿✿ This demonstration repository is under constructions, sound examples, visualizations and additional results from the final experiment will be uploaded, as well as codes and plugin implementation after the reviewing process.


## ✿ ABSTRACT

*Deep generative neural networks* have thrived in the field of computer vision, enabling unprecedented intelligent image processes. Yet the results in audio remain less advanced and many applications are still to be investigated. Our project targets real-time sound synthesis from a reduced set of high-level parameters, including semantic controls that can be adapted to different sound libraries and specific tags. These generative variables should allow expressive modulations of target musical qualities and continuously mix into new styles.

To this extent we train *auto-encoders* on an orchestral database of individual note samples, along with their intrinsic attributes: note class, *timbre domain* (an instrument subset) and *extended playing techniques*. We condition the decoder for explicit control over the rendered note attributes and use *latent adversarial training* for learning expressive style parameters that can ultimately be mixed. We evaluate both generative performances and correlations of the attributes with the latent representation. Our ablation study demonstrates the effectiveness of the *musical conditioning* mechanisms. 

The proposed model generates notes as magnitude spectrograms from any probabilistic latent code samples, with expressive control of orchestral timbres and playing styles. Its training data subsets can directly be visualized in the 3-dimensional latent representation. Waveform rendering can be done offline with the *Griffin-Lim algorithm*. In order to allow real-time interactions, we fine-tune the decoder with a pretrained magnitude spectrogram inversion network and embed the full waveform generation pipeline in a *plugin*. Moreover the encoder could be used to process new input samples, after manipulating their latent attribute representation, the decoder can generate sample variations as an *audio effect* would. Our solution remains rather light-weight and fast to train, it can directly be applied to other sound domains, including an *user's libraries* with *custom sound tags* that could be mapped to specific generative controls. As a result, it fosters creativity and intuitive audio style experimentations.

## ✿ Additional visualizations

**Test spectrogram reconstructions**

We plot log-scaled input mel-spectrograms and reconstructions. We only display test set notes to assess the quality and generalization power of the models.

**Latent spaces and adversarial training**

We plot the learned latent representations of different models and data subsets. 3-dimensional scatter plots are the raw latent coordinates, 2-dimensional are done with t-SNE (t-distributed Stochastic Neighbor Embedding). We use colors to render different attribute subsets: semitones, octaves and styles (playing technique or timbre domain).

For WAE-Fader models, we as well show the evolution of the latent representation throughout the model training with adversarial latent classification. In this setting, the Fader latent discriminator tries to classify style attributes from the non-conditional encoder output **z**. In turn and after the **α-warmup** has started, the encoder tries to fool the discriminator so that its latent representation cannot be properly classified. It encourages an attribute-free latent code and push the decoder to learn its conditioning.

## ✿ Sound examples

We may also upload on soundcloud as github audio streaming is often slow.

**Test set reconstructions GLA and MCNN**

We give test set reconstructions inverted to waveform with either GLA (iterative) or MCNN (feed-forward and realtime capable) to allow for individual listening and evaluation of the current audio-qualities we achieved.

**Random conditional note generations with WAE-style and WAE-Fader**

We give some random note samples, that were generated in the same way as for the note-conditional generative evaluation. Given a random latent point sampled from the model prior and random note targets with octave either in {3-4} (common to all instrument tessitura) or in {0-8} (full orchestral range), we decode with each model's style attribute and invert to waveform with either GLA or MCNN. Here we select the categorical style classes that correspond to each training data subset.

We can see that accordingly to the evaluations of WAE-style, it does not render meaningful audio variations with respect to the target style attributes (both for playing techniques and timbres). Whereas, WAE-Fader produces consistent sound variaitons that express the desired target styles. It demonstrates its ability to efficiently learn and generalize the conditional controls to random samples from the prior, as validated by our evaluations.

**Expressive style and timbre synthesis**
