#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mountain_car
import plot_tools
import batch_learners

# R1: The value of an absorbing state i.e 
# the return if starting from an absorbing state 
# is 0

# R2: Once the end of an episode has been reached
# set gamma to 1-flag, with flag = 1 when we're in 
# an absorbing state

# R3: (x,v) = (0,0)

# R4: Plotting dataset

sim = mountain_car.simulator()

# for i in range(1, 5):
#     n_transitions = 100*i
#     dataset = sim.gen_dataset(n_transitions)
#     plot_tools.plot_transitions(dataset[0],
#                                 dataset[1],
#                                 dataset[2],
#                                 './figs/{}_transitions'.format(n_transitions))

# Generating transitions
n_transitions = 30000
dataset = sim.gen_dataset(n_transitions)
plot_tools.plot_transitions(dataset[0],
                            dataset[1],
                            dataset[2],
                            './figs/transitions/{}_transitions'.format(n_transitions))
# R5: FittedQ
fitQ = batch_learners.fittedQ()

n_iter = 150
res = []
for k in range(n_iter):
    fitQ.update(dataset[0], dataset[1], dataset[2], dataset[3], dataset[4])
    plot_tools.plot_val_pol(fitQ, './figs/fittedQ/val_pol_iter_{}'.format(k))
    traj, cmpt = sim.sample_traj(fitQ)
    res.append(cmpt)
    plot_tools.plot_traj(traj, './figs/fittedQ/traj_iter_{}'.format(k))
plot_tools.plot_perf(res, './figs/fittedQ/perf')

# R5: LSPI
lspi = batch_learners.LSPI()

n_iter = 5
res = []
for k in range(n_iter):
    lspi.update(dataset[0], dataset[1], dataset[2], dataset[3], dataset[4])
    plot_tools.plot_val_pol(lspi, './figs/LSPI/val_pol_iter_{}'.format(k))
    traj, cmpt = sim.sample_traj(fitQ)
    res.append(cmpt)
    plot_tools.plot_traj(traj, './figs/LSPI/traj_iter_{}'.format(k))
plot_tools.plot_perf(res, './figs/LSPI/perf')