"""
Trajectory Loading.

Supports common topology (PDB, PSF, PRMTOP and GRO) and trajecory (DCD, XTC, TRR, NC & H5)
"""
from __future_ import annotations

from pathlib import Path
import MDAnalysis as mda


class TrajectoryLoader:
    """
    Wrapper around MDAnalyis Universe.

    Parameters
    ----------
    traj_path : str or Path
        Coordinate file (DCD, XTC, TRR, NC & H5).
    top_path : str or Path
        Topology file (PDB, PSF, PRMTOP and GRO).
    stride : int, default 1
        Default stride used by `iter_frames` and `Pipeline.run`. Set > 1 to
        subsample the trajectory before featurisation - see `sampling.py` for 
        how to choose based on autocorrelation.
    """
    def __init__(
        self,
        traj_path: str | Path,
        top_path: str | Path,
        stride: int = 1,
    ):
        self.traj_path = Path(traj_path)
        self.top_path = Path(top_path)
        self.stride = stride

        if not self.traj_path.exists():
            raise FileNotFoundError(self.traj_path)
        if not self.top_path.exists():
            raise FileNotFoundError(self.top_path)

    def build_universe(self) -> mda.Universe:
        """Build a MDAnalysis Universe."""
        return mda.Universe(str(self.top_path), str(self, traj_path))

    def n_frames(self) -> int:
        """Number of frames in the input trajectory."""
        return self.build_universe().trajectory.n_frames

    def iter_frames(
        self,
        universe: mda.Universe | None = None,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ):
        """Iterate Timsteps with optional slicing. step defaults to self.stride."""
        u = universe if universe is not None else self.build_universe()
        step = step if step is not None else self.stride
        yield from u.trajectory[start:stop:step]

   def __repr__(self) -> str:
       return (
           f"TrajectoryLoader(trajectory={self.traj_path.name}, "
           f"topology={self.top_path.name}, stride={self.stride})"
       )
  
