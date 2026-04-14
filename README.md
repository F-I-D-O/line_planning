This repository contains line planning code for the MoD-aware line selection problem. Some of the code for candidate line generation and line planning is based on the paper: 'Real-Time Approximate Routing for Smart Transit Systems' URL: <https://arxiv.org/abs/2103.06212>



# Input Files
The input files are organized in the same way as in the [DARP instances project](https://github.com/aicenter/Ridesharing_DARP_instances). There are separate directories for:

- **Instance**: configuration of the problem instance, demand, travel time model, candidate lines, etc., and
- **Results**: experiment directory with experiment/method config and all result and log files.

Both results and instance configurations are `yaml` files, with same structure as in the [DARP instances project](https://github.com/aicenter/Ridesharing_DARP_instances). Only fields specific to line planning are described below in this readme, for others see the [DARP instances project](https://github.com/aicenter/Ridesharing_DARP_instances).

## Instance Files

### New configuration fields

- `lines`: path to the candidate-lines file.


### Candidate Lines Files
Candidate lines files are text files where each line represents a single candidate line. The line is a sequence of node IDs of bus stops that form a potential transit route.

### Preprocessing cache
Computed trip-option preprocessing (CSV) is stored under a ``preprocessing`` folder next to the instance ``config.yaml`` (i.e. ``<instance_directory>/preprocessing/``). The same cache is reused across experiments that point at the same instance when demand, candidate-lines path, and detour match.


## Experiment Files

### New configuration fields

- `mass_transport`: mass transit model parameters.
    - `capacity`: capacity of the MT vehicle.
    - `maximum_detour`: maximum detour for a passenger on MT over the shortest path. Trip options with a detour greater than this value are not considered. It is a relative value greater than 1, i.e. a value of 1.5 means that the passenger is allowed to be 50% longer than the shortest path.
    - `cost_coefficient`: scales line operating cost in the ILP (default `1`). Line cost terms use this coefficient times route length (and frequency where applicable).
    - `max_frequency`: maximum per-route frequency / replication cap for the budgeted ILP formulations (default `1`).

- `solver`: solver driver configuration.
    - `method` (required): one of `approximation`, `ilp`, `ilp_with_mod_costs`, `ilp_with_empty_trips`, `non_budget_ilp`.
    - `time_limit` (seconds): Gurobi time limit (default `86400`).
    - `approximation_subproblem_method` (optional int): Gurobi `Method` for each column-generation sub-MIP when `method` is `approximation`; omit or `0` for default.

- `budget` (optional): budget for the line-planning ILP. Depending on the solution method, MT only or both MT and MoD are restricted by this budget.

Supported `solver.method` values:

- `approximation`: column-generation + randomized rounding.
- `ilp`: budgeted line-planning ILP (baseline).
- `ilp_with_mod_costs`: ILP with MoD cost model (stage 1).
- `ilp_with_empty_trips`: ILP with empty-vehicle trips (stage 2).
- `non_budget_ilp`: MoD-aware route-aggregated ILP without a line budget (same formulation as in `scripts/MoD-aware_line_selection.py`).
