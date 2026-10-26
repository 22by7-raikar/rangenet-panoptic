"""Training-free instance segmentation by Euclidean clustering.

Turns any semantic segmentation into a panoptic one: points of each thing
class are grouped into connected components, where two points are connected
when they are closer than a class-specific radius. This is the classic
"semantics + clustering" baseline; it needs no training and no instance
labels, and fails where objects of one class touch (parked cars, crowds).
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def euclidean_clusters(xyz: np.ndarray, sem: np.ndarray, things: list[int],
                       radius: float | dict[int, float] = 0.5,
                       min_points: int = 1) -> np.ndarray:
    """Instance ids from connected components of each thing class.

    @param xyz         points, (N, 3)
    @param sem         semantic class per point, (N,)
    @param things      thing classes to cluster
    @param radius      linking distance in metres, one value or {class: value}
    @param min_points  components smaller than this get instance 0
    @return instance id per point, (N,), unique within the scan, 0 for stuff
    """
    inst = np.zeros(len(xyz), dtype=np.int64)
    next_id = 1
    for cls in things:
        idx = np.flatnonzero(sem == cls)
        if idx.size == 0:
            continue
        r = radius.get(cls, 0.5) if isinstance(radius, dict) else radius
        pairs = cKDTree(xyz[idx]).query_pairs(r, output_type="ndarray")
        graph = coo_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])),
                           shape=(idx.size, idx.size))
        n, comp = connected_components(graph, directed=False)
        sizes = np.bincount(comp, minlength=n)
        keep = sizes >= min_points
        ids = np.zeros(n, dtype=np.int64)
        ids[keep] = np.arange(next_id, next_id + keep.sum())
        next_id += int(keep.sum())
        inst[idx] = ids[comp]
    return inst
