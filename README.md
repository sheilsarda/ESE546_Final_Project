# Few-Shot Meta Q-Learning with Reptile

Rahul Maganti, Sheil Sarda, Jason Xian

## ESE 546 Final Project 

Deep neural networks work well only when there is a lot of training data present. Moreover, training these networks is often slow. Human intelligence, on the other hand, learns quickly and extrapolates well only from only a small number of data points. Learning in a low-data regime remains a key challenge for Deep Learning. In this paper, we wish to quantitatively validate the effectiveness of few-shot learning in the context of Reinforcement Learning. By testing the Reptile algorithm in OpenAI's CartPole simulation environment using a Deep Q-Network model, we believe that the results obtained from this context will enhance the studies shown on few-shot learning approaches.

Our report for this project can be found [here](ESE546_Final_Report.pdf).

## Install the OpenAI gym environment:

`pip install gym`

Next, find where gym was installed: `pip where gym`. Locate the directory `gym/gym/envs/classic_control` and copy the code from [our cartpole file](setup_environment/cartpole.py) into your file.

Finally, change your directory into `gym/gym/envs` and locate `__init__.py`. Copy the code from [our init file](setup_environment/__init__.py) into your file.


## Run code

Depending on which version you want to run:

`python main.py`

## Misc.

The file `reptile_layout.py` is a sketch of how the algorithm works.
