# Dynamic Obstacle Avoidance using LSTM

## Abstract
For robots and self driving cars to move in human
surroundings the robots should be able to predict the trajectory
to decide its own path and motion. For this reason the problem
of trajectory prediction becomes important. Working on this
problem of trajectory prediction of pedestrians for self driving
scenario, we have compared different models of LSTM and GRU
to show what type of model works best for this problem. We
have compared 4 models- Vanilla LSTM, Social LSTM, OLSTM,
and GRU to show their comparison for predicting non linear
trajectories of pedestrians in different scenes. We demonstrate
their performance on publically available datasets. Comparing all
these models we show how it is important to take into account the
surroundings of the pedestrians to predict with better accuracy.
We humans change our trajectory based on the obstacles we
encounter, so it is essential that we take the obstacles in the
scene as input to our models to predict accurately.

## Results
| Errors              | Vanilla LSTM | GRU      |Social LSTM | OLSTM  |
| :----:              |    :----:    |  :----:  |   :----:   | :----: |
|Avg train disp err   |0.771         | 0.7029   |0.333       |0.407   |
|Final train disp err |1.597         | 1.2921   |0.5021      |0.7752  |
|Avg test disp err    |1.2648        |1.3957    |1.2176      |1.236   |
|Final test disp err  |2.755         |2.6529    |2.543       |2.643   |

More results and graphs are available in [report](./report.pdf)
## Conclusion and Discussion
We have in our work presented different versions of LSTMs
that can predict human trajectory. By comparing the results of
all these versions, we can see how social LSTM gives the best
result. This is expected as the social LSTM takes into account
the neighboring trajectories and use the social pooling layer to
include the social interaction parameters humans show while
walking.

We also see how GRU gives similar results as it shares
the same architecture but because it is less computationally
expensive, it can be used to run on self driving car or any robot
in real time to predict trajectories. LSTM had marginally better
performance, though a longer run time. GRU performed very
similar to LSTM, but much shorter run time since the GRU cell
has lesser number of gates.GRU uses less training parameter
and therefore uses less memory and executes faster than LSTM
whereas LSTM is more accurate on a larger dataset. One
can choose LSTM if we are dealing with large sequences
and accuracy is concerned, GRU is used when we have less
memory consumption and want faster results.

Our models are able to predict non linear trajectories with
reasonable accuracy. The results can be made better with more
data obtained in varying scenery to make our training more
robust. More data will be able to make our predictions for
longer trajectories better.

We can interpret from our results that predicting only based
on few datapoints will not help us in learning the non linear
predictions to far length into the future. We tried to add the
penalty in function of the distance to neighboring pedestrian in
our loss function believing that it can help our models improve
their collision avoidance properties, but we were not able to get
improved results. We also learned through our implementations that how
important the distribution of our dataset is. We first tried only
using a subset of the dataset, only using ETH dataset. That
did not give us good results because our models overfit the
dataset. We also had to play around with number of epochs to
train our models on, as they were not giving very good results
with low number of epochs which have been used in some
papers.

As possible extensions to our work, we are interested in
exploring how the inclusion of different agents in the same
scene affects the behavior of our models. We have only trained
and tested on scenes which contain only pedestrians. Inclusion
of cars, bicycles, skateboards, etc in the scene will make our
problem more complex. While considering future trajectories,
we should also consider the scenery of our agents. People will
respond to the objects in their surroundings that will act like
obstacles. This scene information should also be taken as input
to our models to predict trajectory in a better way.