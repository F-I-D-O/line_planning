This repository contains line planning code for the MoD-aware line selection problem. Some of the code for candidate line generation and line planning is based on the paper: 'Real-Time Approximate Routing for Smart Transit Systems' URL: https://arxiv.org/abs/2103.06212



# Input Files
The input files are organized in the same way as in the [DARP instances project](https://github.com/aicenter/Ridesharing_DARP_instances). There are separate directories for:

- **Instance**: configuration of the problem instance, demand, travel time model, candidate lines, etc., and
- **Results**: experiment directory with experiment/method config and all result and log files.

Both results and instance configurations are `yaml` files, with same structure as in the [DARP instances project](https://github.com/aicenter/Ridesharing_DARP_instances). Only fields specific to line planning are described below in this readme, for others see the [DARP instances project](https://github.com/aicenter/Ridesharing_DARP_instances).

## Instance Files

### New cofiguration fields

- `lines`: path to the candidate-lines file.


### Candidate Lines Files
Candidate lines files are text files where each line represents a single candidate line. The line is a sequence of node IDs of bus stops that form a potential transit route.


## Experiment Files

### New cofiguration fields

- `mass_transport`: object collecting mass transport configuration.
    - `capacity`: the capacity of the MT vehicle.
    - `maximum_detour`: the maximum detour for a passenger of the MT vehicle over the shortest path. Trip options with a detour greater than this value are not considered. It is a relative value greater than 1, i.e. a value of 1.5 means that the passenger is allowed to be 50% longer than the shortest path.





