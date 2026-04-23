"""Minimal dill compatibility shim for this repo's training scripts."""
import pickle

load = pickle.load
loads = pickle.loads
dump = pickle.dump
dumps = pickle.dumps


def extend(*args, **kwargs):
    return None
