# GANs for people re-identification projects

One track of the semantic vision project is connected to videoanalytics with two subtracks -
1) people re-identification on cameras with non-overlapping views
2) semantic search for people in videos from cameras.

Here, the first subtrack is discussed.


## Experiments to be done


  0. We should study the possibility of constructing a useful generative model of people
      - [ ] The first simplest experiment will be to train GANs, e.g., DC-BEGANs or other type of GANs with available implementation and good quality, on datasets with detected/tracked humans (not necessarily re-id datasets, but they can also be used here since we are familiar with them)
      - [ ] Analyze the quality of generated images. Is it reasonable?
      - [ ] Compare quality of non-Info-GAN with (DC-)(BE-)Info-GAN

  1. Analyze disentangled features
      - [ ] Apply semantical decomposition (SD-) to DC-BEGANs: first part of the latent code should correspond to person ID, second part should correspond to camera ID, third part will include rest features.
      - [ ] Introduce SD- to (DC-)(BE-)Info-GANs, i.e. impose supervised semantic decomposition to Info-GANs
      - [ ] Analyze the results of differetn models and compare them
