# Semantic Vision for OpenCog

This is a repository hosting public discussion, documentation and milestones of the semantic vision research initiative within OpenCog. Please see the [Wiki](https://github.com/opencog/semantic-vision/wiki) for more information! It is starting as of September 2017 and we are adding research material, code used for experimentation and results as we go along. Stay tuned for more!

## Abstract

We propose a novel architecture for recognizing, generating and reasoning about patterns in perceptual data.   

The first core component of the architecture is SynerGAN, an extension of the InfoGAN methodology to incorporate symbolic probabilistic logic and symbolic pattern mining, alongside subsymbolic neural net learning. In SynerGAN, each of the players in the game underlying InfoGAN, includes both symbolic and subsymbolic learning components, collaborating together to play their part in the game. The symbolic components play the role of the structured latent variables in InfoGAN, and are carrying out probabilistic reasoning and pattern mining, in a way that interacts appropriately with the subsymbolic (neural) components.

The second core component of the architecture is compositionality: in order to comprehend a complex perception, a number of different SynerGAN networks are applied to different portions of the perceptual data in a judicious way, in which each SynerGAN network may potentially act upon other SynerGAN networks as well as on raw data.  The composition of multiple SynerGAN networks is done in a way that respects the semantics of the probabilistic latent variables of the networks.



## Goals and Milestones

This is an outline of our major milestones and progress. A more detailed description can be found on the [Wiki](https://github.com/opencog/semantic-vision/wiki/Implementation-Milestones). These will be formulated with greater granularity once we have proceeded a bit.

  0. Preliminary experiments around different implementations of GAN and InfoGAN
  1. [Probabilistic Network InfoGAN](https://github.com/opencog/semantic-vision/wiki/ProNetInfoGAN)
  2. [SynerGAN](https://github.com/opencog/semantic-vision/wiki/About-the-SynerGAN-architecture)
  3. [Compositional SynerGAN](https://github.com/opencog/semantic-vision/wiki/Composition-of-SynerGAN)



## Collaboration

This is an open source, geographically decentralized research project. We have several core contributors spanning multiple timezones. Any contribution from outside such as discussion, experimentation results or code is always welcome! We use several mechanisms to coordinate the project between the different parties:

  * Issues are the main form of discussion and will be used for questions, proposals and support with using our code.

  * Any contribution to existing code or new implementations will be discussed and merged as pull requests.

  * The Wiki will be used to keep track of all research related materials and contains some material from our initial project proposal.

  * Projects can be used by indivudal team members to keep track of private projects but may also be used to track collaborative efforts in more detail in the future.
