import os
import sys
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14, 8]

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)
from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import Tracker, get_dataset, trackerlist

############ Packed trackingnet results ############
from pytracking.util_scripts import pack_trackingnet_results
pack_trackingnet_results.pack_trackingnet_results('feast', 'feast', 0, 'FEAST')
