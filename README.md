This repository contains line planning code for the MoD-aware line selection problem. Some of the code for candidate line generation and line planning is based on the paper: 'Real-Time Approximate Routing for Smart Transit Systems' URL: <https://arxiv.org/abs/2103.06212>



# Input Files
The input files are organized in the same way as in the [DARP instances project](https://github.com/aicenter/Ridesharing_DARP_instances). There are separate directories for:

- **Instance**: configuration of the problem instance, demand, travel time model, candidate lines, etc., and
- **Results**: experiment directory with experiment/method config and all result and log files.

Both results and instance configurations are `yaml` files, with same structure as in the [DARP instances project](https://github.com/aicenter/Ridesharing_DARP_instances). Only fields specific to line planning are described below in this readme, for others see the [DARP instances project](https://github.com/aicenter/Ridesharing_DARP_instances).

## Instance Files

### New configuration fields

- `lines`: path to the candidate-lines file. If omitted or empty, defaults to `lines.txt` in the same directory as the instance `config.yaml`.


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
    - `line_mod_aggregate_prune`: if true, prune line-mod aggregate trip options. Default is false.

- `solver`: solver driver configuration.
    - `method` (required): one of `approximation`, `ilp`, `ilp_with_mod_costs`, `ilp_with_empty_trips`, `non_budget_ilp`.
    - `time_limit` (seconds): Gurobi time limit (default `86400`).
    - `approximation_subproblem_method` (optional int): Gurobi `Method` for each column-generation sub-MIP when `method` is `approximation`; omit or `0` for default.
    - `rejection_cost`: cost of rejecting a passenger. Applicable only when `method` is `non_budget_ilp`. Default is 0, which means that all passengers must be served.
    - `use_request_line_valid_inequalities`: if true, add valid linking inequalities `x_{\rho r} \leq y_\rho` for each route ρ and request r (only when `method` is `non_budget_ilp`). Default is false.
    - `reuse_model`: if true, reuse the ILP model. This only affects the MoD-aware line selection ILP, and it makes sense only if we solve the same instance multiple times. Default is true.

- `budget` (optional): budget for the line-planning ILP. Depending on the solution method, MT only or both MT and MoD are restricted by this budget.

Supported `solver.method` values:

- `approximation`: column-generation + randomized rounding.
- `ilp`: budgeted line-planning ILP (baseline).
- `ilp_with_mod_costs`: ILP with MoD cost model (stage 1).
- `ilp_with_empty_trips`: ILP with empty-vehicle trips (stage 2).
- `non_budget_ilp`: MoD-aware route-aggregated ILP without a line budget (same formulation as in `scripts/MoD-aware_line_selection.py`).

## MoD-aware Line Selection Experiment Files
MoD-aware line selection experiment files are `yaml` files used to configure the iterative line selection process. Because each iteration encompases solving both the line selection and DARP problem, the experiment is an extension of the [Experiment Files](#experiment-files) described above. The extension fields are:

- `darp`: DARP configuration.
    - `benchmark_executable`: path to the DARP-benchmark binary.
    - `transfer_delay`: the delay in seconds for the transfer between the line and the DARP vehicle. Currently, it only affects the departure/arrival times for MoD, not the cost.
- `initial_mod_cost_scale`: the scale factor for the MoD cost received from the line instance. This is used only in the first iteration.
- `iterations`: the number of iterations to run.
- `mod_cost_recomputation`: how MoD costs are updated after each DARP solve.
    - `strategy`: recomputation strategy (see `scripts/MoD-aware_line_selection.py`).
    - `smoothing` (optional): post-recomputation smoothing applied before writing costs back into the line instance.
        - `strategy`: e.g. `none` or `under_relaxation`.
        - `under_relaxation_alpha`: blend parameter in \([0, 1]\) when using `under_relaxation`.


### Cost Recomputation Strategies
The cost recomputation strategies are:

- `aggregate_original_requests`: aggregate the MoD costs for the original requests.
- `rescale_avg_cost_per_traveltime_and_request`: rescale the MoD costs by the average cost per travel time and request.

### Cost Smoothing Strategies
The cost smoothing strategies are:

- `none`: no smoothing.
- `under_relaxation`: under-relaxation smoothing.
