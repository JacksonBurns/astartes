"""
The Sphere Exclusion clustering algorithm.

This implementation draws heavily from this blog post on the RDKit blog,
though abstracted to work for arbitrary feature vectors:
http://rdkit.blogspot.com/2020/11/sphere-exclusion-clustering-with-rdkit.html
"""

from astartes.samplers import AbstractSampler


class SphereExclusion(AbstractSampler):
    def __init__(self, *args):
        super().__init__(*args)

    def _sample(self):
        pass
