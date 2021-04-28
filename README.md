# SQM basis 

Human visual perception is rather discrete than continuous. Drissi-Daoudi and colleagues used SQM (sequential meta-contrast) paradigm and figured out that there is discrete time windows (~450ms) for visual temporal integration. Here with computational models, we examine what makes us perceive discrete way and what is the source of opening and closing the perceptual time windows. 

We combined two distinct projects, SQM_frame prediction and SQM_discreteness proejcts, together in the current project. 

- SQM_frame prediction: https://github.com/albornet/sqm_models
- SQM_discreteness: https://github.com/Ohyeon5/SQM_discreteness

# Network architecture

Networks must be structured as follows: a convolutional module, a encoder and a decoder.

# Workflow

## Training

### Training the encoder

First, the whole network is trained on the task of classifying human hand gestures, with the objective of getting the encoder to learn time dependency.

`python main.py rc=phase1`

### Training the convolutional module and the decoder

Then the encoder is frozen, and only the convolutional module and the decoder are trained on a vernier classification task.

`python main.py rc=phase2`

## Testing

The network is tested on the standard SQM paradigm.

`python test.py`

### References
>>Drissi-Daoudi, L., Doerig, A., & Herzog, M. H. (2019). Feature integration within discrete time windows. Nature communications, 10(1), 1-8. https://doi.org/10.1038/s41467-019-12919-7

>> Herzog, M. H., Drissi-Daoudi, L., & Doerig, A. (2020). All in Good Time: Long-Lasting Postdictive Effects Reveal Discrete Perception. Trends in Cognitive Sciences. https://doi.org/10.1016/j.tics.2020.07.001