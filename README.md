[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"


## Introduction

In this project, we train an agent to navigate (and collect bananas!) in a large, square world. We train the agent using the DQN algorithm, which is described in the [report](Report.md).

![Trained Agent][image1]

### Environment

The task is episodic, and in order to solve the environment, the agent should get an average score of +13 over 100 consecutive episodes.

#### Reward

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

#### States

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  

#### Actions

Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

### Environment Setup

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in this project's root, and unzip (or decompress) the file.

3. Install the needed python packages (Best to create a [virtual environment](https://virtualenv.pypa.io/en/latest/) to contain them first!)

```
pip install -r requirements.txt
```

### Training

Run the cells in the notebook [Training.ipynb](Training.ipynb) in order to train the agent. If it reaches the threshold set in the `train_agent` function, then it will save the model weights in `solved_model.pt`.

### Report

Additional details about the implementation of the algorithm can be found in [Report.md](Report.md).
