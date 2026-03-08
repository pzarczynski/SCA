import numpy as np
from numpy.typing import NDArray

def fused_select_poly(
    x: NDArray[np.float32], 
    y: NDArray[np.uint64],
    n_classes: int,
    select: int,
    k: int,
) -> tuple[list[tuple[np.float32, int, int]],
           NDArray[np.float32], NDArray[np.float32]]: ...

def fused_transform_poly(
    x: NDArray[np.float32], 
    means: NDArray[np.float32],
    scales: NDArray[np.float32],
    ix: NDArray[np.uint64],
    jx: NDArray[np.uint64],
) -> NDArray[np.float32]: ...