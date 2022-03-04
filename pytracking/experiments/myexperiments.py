from pytracking.evaluation import Tracker, get_dataset, trackerlist


def got10k():
    # Run three runs of FEAST on Got-10K dataset
    trackers = trackerlist('feast', 'feast', range(1))

    dataset = get_dataset('got10k_test')
    return trackers, dataset


def otb():
    # Run three runs of FEAST on OTB100 dataset
    trackers = trackerlist('feast', 'feast', range(1))

    dataset = get_dataset('otb')
    return trackers, dataset

def trackingnet():
    # Run three runs of FEAST on TrackingNet dataset
    trackers = trackerlist('feast', 'feast', range(1))

    dataset = get_dataset('trackingnet')
    return trackers, dataset

def uav():
    # Run three runs of FEAST on UAV123 dataset
    trackers = trackerlist('feast', 'feast', range(1))

    dataset = get_dataset('uav')
    return trackers, dataset

def lasot():
    # Run three runs of FEAST on LaSOT dataset
    trackers = trackerlist('feast', 'feast', range(1))

    dataset = get_dataset('lasot')
    return trackers, dataset

def tpl():
    # Run three runs of FEAST on TCL128 dataset
    trackers = trackerlist('feast', 'feast', range(1))

    dataset = get_dataset('tpl')
    return trackers, dataset

def nfs():
    # Run three runs of FEAST on NFS dataset
    trackers = trackerlist('feast', 'feast', range(1))

    dataset = get_dataset('nfs')
    return trackers, dataset
