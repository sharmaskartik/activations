# learning to use pytorch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import reveal
import reveal.experiments.performance as rf
import reveal.parameters.constants as consts
import data_utils.io.auto_mpg as mpg
import data_utils.io.storage_constants as sc
import reveal.util.io as io
from reveal.structures.loss_experiment_structure import *
from reveal.graphics.plot.loss_plotter import *

import json
import time

dataPath = sc.DATA_SETS_lOCATION+"auto_mpg/auto-mpg.data"

x, t, x_names, t_names = mpg.make_mpg_data(dataPath)

lc = rf.LossComparision()

lc.params['repetitions'] = 5
lc.params['epochs'] = 30
lc.params['batch_size'] = 20
lc.params['problem_type'] = consts.PROBLEM_TYPE_REGRESSION
lc.params['network_type'] = consts.NETWORK_TYPE_FEEDFORWARD
lc.params['net_structures'] = [[4000, 5000,4000,4000,4000]]
lc.params['activation_fs'] = ["relu", "tanh"]
lc.params['loss_function'] = nn.MSELoss()
lc.params['optimizer'] = []
lc.params['use_cuda'] = False
lc.params['verbosity'] = consts.VERBOSE_MED_INFO

results = lc.compare_loss(x, t)

lp = LossPlotter(results)
lp.params["train_label"] = "Training MSE"
lp.params["test_label"] = "Testing MSE"
lp.params["y_axis_label"] = "MSE"


#fig, ax = lp.get_plots(lp.TYPE_ALL_REPETITIONS, reps = 200, alpha = 0.4, show = True)

# fig, ax = lp.get_plots(lp.TYPE_MEAN_ACTIVATIONS_TOGETHER, alpha = 0.4, show = True)
# fig, ax = lp.get_plots(lp.TYPE_MEAN_ACTIVATIONS_SEPARATE, alpha = 0.4, show = True)
#fig.show()
s_t = time.time()
#print(results.read_me)
# results.save("mean.pickle")
#
#
# a = io.load("mean.pickle")
# a = LossExperimentResults(obj = a)
# a.print_results()
