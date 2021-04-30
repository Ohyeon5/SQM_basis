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

Two datasets are available: one with 2 hand gestures (swipe left & swipe right) and one with 5 hand gestures. By default, the one with 2 hand gestures is selected. To instead select the one with 5 hand gestures, execute the following:

`python main.py rc=phase1 rc.train_data_artifact=hand_gestures_train_5:latest`

Training is performed with automated early stopping: when the loss stops going down, training stops. However, if you want to set a maximum number of epochs, you can do so by appending `rc.n_epochs=<number of epochs>` to the command.

### Training the convolutional module and the decoder

Then the encoder is frozen, and only the convolutional module and the decoder are trained on a vernier classification task.

`python main.py rc=phase2`

## Testing

The network is tested on the standard SQM paradigm. By default, a video sequence has 13 frames, and all pro-vernier and anti-vernier conditions are tested. For each condition, several sequences are tested, where the two parameters which vary are the vernier starting position and the vernier size.

`python test.py`

## Viewing results

The terminal in which you run the training or testing commands will display results as they come. However, to get the big picture, or to visualize results at a later time, you can use the Weights & Biases [dashboard](https://wandb.ai/lpsy_sqm/lr-vernier-classification) (provided you are in the lpsy_sqm team and are logged in to Weights & Biases).

### References
>>Drissi-Daoudi, L., Doerig, A., & Herzog, M. H. (2019). Feature integration within discrete time windows. Nature communications, 10(1), 1-8. https://doi.org/10.1038/s41467-019-12919-7

>> Herzog, M. H., Drissi-Daoudi, L., & Doerig, A. (2020). All in Good Time: Long-Lasting Postdictive Effects Reveal Discrete Perception. Trends in Cognitive Sciences. https://doi.org/10.1016/j.tics.2020.07.001