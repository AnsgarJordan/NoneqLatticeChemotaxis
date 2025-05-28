import argparse

parser = argparse.ArgumentParser(description="ising_neq_v6")
parser.add_argument("--N", type=int, default=6, help="lattice size is N*N")
parser.add_argument("--J", type=float, default=1.18, help="coupling strength")
parser.add_argument("--k2", type=float, default=1.0, help="rate k2 sets the time scale")
parser.add_argument("--k3", type=float, default=0.5, help="rate k3")
parser.add_argument(
    "--epsilon", type=float, default=0.0, help="epsilon = exp(-deltaG/3)"
)

parser.add_argument(
    "--activity-threshold",
    type=float,
    default=0.02,
    help="Threshold for balancing the activities",
)
parser.add_argument(
    "--prominence-threshold",
    type=float,
    default=0.01,
    help="prominence threshold for identifying the peaks",
)

parser.add_argument(
    "--step-thermalize",
    type=int,
    default=10000000,
    help="steps to thermalize the system; need to be thrown away for computing the switching time",
)
parser.add_argument(
    "--max-step-balancing",
    type=int,
    default=50000000,
    help="max number of KMC steps for finding the balanced k1 (per step in the binary search)",
)
parser.add_argument(
    "--max-step-evaluating",
    type=int,
    default=90000000,
    help="max number of KMC steps for calculating the final dwell and switching time (per k3 value)",
)
parser.add_argument("--random-seed", type=int, default=97, help="random seed for KMC")
parser.add_argument(
    "--dataname", type=str, default="data_demo", help="directory for output"
)
