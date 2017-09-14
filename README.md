# Semantic Vison within OpenCog

This is the official repository of the semantic vision research initiative within OpenCog. As of September 2017 it is just starting and we are slowly adding research material, code used for experimentation and results.

## Abstract

We propose a novel architecture for recognizing, generating and reasoning about patterns in perceptual data.   

The first core component of the architecture is SynerGAN, an extension of the InfoGAN methodology to incorporate symbolic probabilistic logic and symbolic pattern mining, alongside subsymbolic neural net learning. In SynerGAN, each of the players in the game underlying InfoGAN, includes both symbolic and subsymbolic learning components, collaborating together to play their part in the game. The symbolic components play the role of the structured latent variables in InfoGAN, and are carrying out probabilistic reasoning and pattern mining, in a way that interacts appropriately with the subsymbolic (neural) components.

The second core component of the architecture is compositionality: in order to comprehend a complex perception, a number of different SynerGAN networks are applied to different portions of the perceptual data in a judicious way, in which each SynerGAN network may potentially act upon other SynerGAN networks as well as on raw data.  The composition of multiple SynerGAN networks is done in a way that respects the semantics of the probabilistic latent variables of the networks.



## Goals and Milestones

This is an outline of our major milestones and progress towards them. A more detailed description can be found on the Wiki in [[Implementation Steps]], These will be formulated with greater granularity once we have proceeded a bit.

### Milestone 0: Merging of BEGAN and InfoGAN
  - [ ] Work towards our reference implementation of BEInfoGAN in the `implementations` subfolder.
  - [ ] Apply BEInfoGAN to CelebA dataset and achieve reasonable experimental results with our implementation.

### Milestone 1: Probabilistic Network InfoGAN

Here we want to research the loss function needed to enforce structure from arbitrary complex probability distributions on our BEInfoGAN to make ProNetInfoGAN. More discussion on this will follow soon.

### Milestone 2: SynerGAN

Here we will tackle first experimentation with ProNetInfoGANs in combination with PLN to form SynerGAN.

### Milestone 2: Compositional SynerGAN

Introducing clustering into the mix to work towards compositional use of SynerGAN modules.



## Collaboration

This is an open source, geographically decentralized research project. We have several core contributors spanning multiple timezones. Any contribution from outside such as discussion, experimentation results or code is always welcome! We use several mechanisms to coordinate the project between the different parties:

  * Issues are the main form of discussion and will be used for proposals, asking questions,

  * Any contribution to existing code or new implementations will be discussed and merged as pull requests.

  * The Wiki will be used to keep track of all research related materials and contains some material from our initial project proposal.

  * Projects can be used by teammembers to keep track of private projects but may also be used to track collaborative efforts in more detail.
