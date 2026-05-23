import argparse
import csv
import json
import logging
import re
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from gurobipy import *

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lineplanning.instance import *
from lineplanning.instance_config import (
    load_experiment_yaml,
    load_line_planning_instance_config,
    resolve_instance_config_path,
    resolve_results_dir,
)
import lineplanning.log

EPS = 1.e-5


class NoAssignmentHandling(Enum):
    NO_MT = "no_mt"
    REJECT = "reject"
    RAISE = "raise"


@dataclass(frozen=True)
class LineSelectionSolveResult:
    """
    Unified return type for line-selection ILPs used by ``run_experiment`` and the MoD-aware
    line selection script.

    ``request_assignments`` entries are ``("no_MT", None)``, ``("line", rho)`` with route index
    ``rho``, or ``("rejected", None)``.

    ``line_objective_component`` / ``mod_objective_component`` may be ``None`` when the
    formulation does not expose the same decomposition as the MoD-aware ILP.
    """

    objective_value: float
    run_time_seconds: float
    selected_lines: List[int]
    request_assignments: Tuple[Tuple[str, Optional[int]], ...]
    line_objective_component: Optional[float]
    mod_objective_component: Optional[float]

    def __iter__(self):
        """Allow ``a, b, c, d, e, f = solver.solve_*`` unpacking like the legacy 6-tuple."""
        yield from (
            self.objective_value,
            self.run_time_seconds,
            self.selected_lines,
            self.request_assignments,
            self.line_objective_component,
            self.mod_objective_component,
        )


def get_instance_size_label(demand_file_path: Optional[str]) -> str:
    if demand_file_path:
        demand_file_name = Path(demand_file_path).name
        match = re.search(r"(\d+)_percent", demand_file_name)
        if match:
            return f"{match.group(1)}_percent"
    return "100_percent"


# ``solver.method`` in experiment YAML — exactly one runner branch in ``run_experiment``.
VALID_SOLVER_METHODS = frozenset(
    {
        "approximation",
        "ilp",
        "ilp_with_mod_costs",
        "ilp_with_empty_trips",
        "non_budget_ilp",
        "peak_batch_max_headway_ilp",
    }
)


def _resolve_solver_method(solver_cfg: Dict[str, Any]) -> str:
    """Return ``solver.method``; must be a non-empty string in ``VALID_SOLVER_METHODS``."""
    raw = solver_cfg.get("method")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            f"solver.method is required and must be a non-empty string; "
            f"expected one of {sorted(VALID_SOLVER_METHODS)}"
        )
    method = raw.strip()
    if method not in VALID_SOLVER_METHODS:
        raise ValueError(
            f"Unknown solver.method {method!r}; expected one of {sorted(VALID_SOLVER_METHODS)}"
        )
    return method


def _configure_run_logging(log_path: Path) -> logging.Handler:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    return handler


class LinePlanningSolver:

    def __init__(
        self,
        line_instance,
        time_limit: float = 3600 * 24,
        cost_coefficient: float = 1.0,
        max_frequency: int = 1,
    ):
        self.line_instance = line_instance
        self.time_limit = float(time_limit)
        self.cost_coefficient = float(cost_coefficient)
        self.max_frequency = int(max_frequency)
        self.line_count_total = self.line_instance.nb_lines * self.max_frequency
        self._mod_aware_mip_state: Optional[Dict[str, Any]] = None

    def _export_used_lines(
        self,
        output_dir: Path,
        line_vars,
        line_costs,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "used_lines.csv"
        try:
            with csv_path.open("w", encoding="utf-8") as csv_file:
                csv_file.write("line,frequency,line_cost\n")
                for l, var in line_vars.items():
                    activation = var.X
                    if activation > 0:
                        csv_file.write(
                            f"{l // self.max_frequency},{(l % self.max_frequency) + 1},{line_costs[l]}\n"
                        )
            logging.info("Exported used lines to %s", csv_path)
        except OSError as exc:
            logging.warning("Unable to write used lines CSV %s: %s", csv_path, exc)

    def _export_used_lines_route_agg(
        self,
        output_dir: Path,
        frequency_vars,
        per_route_mt_cost_coeff: list,
    ) -> None:
        """
        Export used_lines.csv for the route-aggregated ILP (§4.1.1): one row per route with
        integer frequency y_ρ and MT cost (sum_{e in ρ} c_e) · y_ρ.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "used_lines.csv"
        try:
            with csv_path.open("w", encoding="utf-8") as csv_file:
                csv_file.write("line,frequency,line_cost\n")
                for rho, var in frequency_vars.items():
                    y_val = var.X
                    if y_val > EPS:
                        f_int = int(round(y_val))
                        line_cost = per_route_mt_cost_coeff[rho] * f_int
                        csv_file.write(f"{rho},{f_int},{line_cost}\n")
            logging.info("Exported used lines (route-aggregated) to %s", csv_path)
        except OSError as exc:
            logging.warning("Unable to write used lines CSV %s: %s", csv_path, exc)

    def _direct_trip_mod_cost(self, passenger_idx: int) -> float:
        """
        MoD cost for the pure-MoD (no_MT) option as stored in ``direct_trip_options``.
        This is what the MoD-aware ILP and exports should use so DARP recomputation can
        change direct costs without rewriting ``dm``.
        """
        return self.line_instance.direct_trip_mod_cost(passenger_idx)

    def update_mod_costs(
        self,
        new_costs: Dict[int, Tuple[float, float]],
        request_assignments: list,
    ) -> None:
        """
        Update the MoD cost estimates in optimal_trip_options based on DARP solution (section 4.3.2).

        For each original request, updates the first_mile_cost and last_mile_cost of the
        assigned travel option (either a line or the no_MT option).

        Args:
            new_costs: Dict mapping original_request_id -> (first_mile_cost, last_mile_cost).
                       For MoD-only requests, last_mile_cost should be 0.
            request_assignments: List of (kind, line_idx) tuples from :class:`LineSelectionSolveResult`.
                                 kind is "no_MT", "line", or "rejected"; for "line", line_idx is the route index ρ.
        """
        for original_request_id, (first_mile_cost, last_mile_cost) in new_costs.items():
            kind, line_idx = request_assignments[original_request_id]

            if kind == "rejected":
                continue

            if kind == "no_MT" or line_idx is None:
                self.line_instance.set_direct_trip_costs(
                    original_request_id,
                    first_mile_cost,
                    last_mile_cost,
                )
            else:
                route_index = line_idx
                self.line_instance.set_trip_mod_cost_on_line(
                    original_request_id,
                    route_index,
                    first_mile_cost,
                    last_mile_cost,
                )

        logging.info("Updated MoD costs for %d requests", len(new_costs))

    def _assignments_from_passenger_vars(
        self,
        passenger_vars,
        no_mt_line_key: Optional[int] = None,
        line_var_is_route_index: bool = False,
        rejection_vars: Optional[Dict[int, Any]] = None,
    ) -> Dict[int, Tuple[str, Optional[int]]]:
        """Extract assigned passengers from Gurobi assignment variables."""
        no_mt_key = self.line_count_total if no_mt_line_key is None else no_mt_line_key
        assignments: Dict[int, Tuple[str, Optional[int]]] = {}

        if rejection_vars is not None:
            for passenger_idx, var in rejection_vars.items():
                if var.X > EPS:
                    assignments[int(passenger_idx)] = ("rejected", None)

        for (line_idx, passenger_idx), var in passenger_vars.items():
            if var.X > 0:
                passenger_idx = int(passenger_idx)
                if assignments.get(passenger_idx) == ("rejected", None):
                    continue
                is_no_mt = line_idx == no_mt_key
                if is_no_mt:
                    assignments[passenger_idx] = ("no_MT", None)
                else:
                    route_index = (
                        int(line_idx)
                        if line_var_is_route_index
                        else line_idx // self.max_frequency
                    )
                    assignments[passenger_idx] = ("line", int(route_index))
        return assignments

    def _resolve_and_export_request_assignments(
        self,
        output_dir: Path,
        request_assignments: Union[
            Dict[int, Tuple[str, Optional[int]]],
            List[Tuple[str, Optional[int]]],
            Tuple[Tuple[str, Optional[int]], ...],
        ],
        no_assignment_handling: NoAssignmentHandling = NoAssignmentHandling.RAISE,
        no_mt_line_key: Optional[int] = None,
    ) -> Tuple[List[Tuple[str, Optional[int]]], pd.DataFrame]:
        """
        Resolve missing passenger assignments, export passenger_assignments.csv, and return
        both the canonical assignment list and the richer internal DataFrame.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "passenger_assignments.csv"
        no_mt_key = self.line_count_total if no_mt_line_key is None else no_mt_line_key

        if isinstance(request_assignments, dict):
            partial = {int(k): v for k, v in request_assignments.items()}
        else:
            partial = {idx: value for idx, value in enumerate(request_assignments)}

        full_assignments: List[Tuple[str, Optional[int]]] = []
        for passenger_idx in range(self.line_instance.nb_pass):
            assignment = partial.get(passenger_idx)
            if assignment is None:
                if no_assignment_handling == NoAssignmentHandling.NO_MT:
                    assignment = ("no_MT", None)
                elif no_assignment_handling == NoAssignmentHandling.REJECT:
                    assignment = ("rejected", None)
                elif no_assignment_handling == NoAssignmentHandling.RAISE:
                    raise ValueError(f"Missing assignment for passenger {passenger_idx}")
                else:
                    raise ValueError(f"Unknown no-assignment handling: {no_assignment_handling!r}")
            kind, route_index = assignment
            if kind == "line" and route_index is None:
                raise ValueError(f"Line assignment for passenger {passenger_idx} has no route index")
            if kind not in ("line", "no_MT", "rejected"):
                raise ValueError(f"Unknown assignment kind for passenger {passenger_idx}: {kind!r}")
            full_assignments.append((kind, int(route_index) if route_index is not None else None))

        rows = []
        for passenger_idx, (kind, route_index) in enumerate(full_assignments):
            if kind == "line":
                assert route_index is not None
                line_repr: Union[int, str] = int(route_index)
                line_index = int(route_index)
                route_repr: Union[int, float] = int(route_index)
                mod_cost = self.line_instance.trip_mod_cost_on_line(passenger_idx, int(route_index))
                is_no_mt = False
            elif kind == "no_MT":
                line_repr = "no_MT"
                line_index = no_mt_key
                route_repr = np.nan
                mod_cost = self._direct_trip_mod_cost(passenger_idx)
                is_no_mt = True
            else:
                line_repr = "rejected"
                line_index = -2
                route_repr = np.nan
                mod_cost = 0.0
                is_no_mt = False
            rows.append(
                {
                    "passenger": passenger_idx,
                    "line_index": line_index,
                    "route_index": route_repr,
                    "line_repr": line_repr,
                    "mod_cost": mod_cost,
                    "is_no_mt": is_no_mt,
                }
            )

        assignments_df = pd.DataFrame(rows)
        if not assignments_df.empty:
            assignments_df.sort_values("passenger", inplace=True)
            assignments_df.reset_index(drop=True, inplace=True)
        else:
            assignments_df = pd.DataFrame(
                columns=["passenger", "line_index", "route_index", "line_repr", "mod_cost", "is_no_mt"]
            )

        try:
            export_df = assignments_df[["passenger", "line_repr", "mod_cost"]].copy()
            export_df.rename(columns={"line_repr": "line"}, inplace=True)
            export_df.to_csv(csv_path, index=False)
            logging.info("Exported passenger assignments to %s", csv_path)
        except OSError as exc:
            logging.warning("Unable to write passenger assignments CSV %s: %s", csv_path, exc)

        return full_assignments, assignments_df

    def _solve_and_export_flows(
        self,
        assignments: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        required_flow = defaultdict(int)
        used_nodes = set()
        for row in assignments.itertuples(index=False):
            line_index = int(row.line_index)
            passenger_idx = int(row.passenger)

            # skip rejected requests
            if line_index < 0:
                continue

            origin = self.line_instance.requests[passenger_idx][0]
            destination = self.line_instance.requests[passenger_idx][1]
            used_nodes.update([origin, destination])

            if bool(row.is_no_mt):
                required_flow[(origin, destination)] += 1
            else:
                route_index = row.route_index
                if pd.isna(route_index):
                    continue
                route_index = int(route_index)
                if not self.line_instance.has_trip_option_on_line(passenger_idx, route_index):
                    continue
                pickup, drop_off = self.line_instance.trip_pickup_dropoff_on_line(
                    passenger_idx,
                    route_index,
                )
                required_flow[(origin, pickup)] += 1
                required_flow[(drop_off, destination)] += 1
                used_nodes.update([pickup, drop_off])

        # debug required flow cost
        C_req = sum(int(self.line_instance.dm[i][j]) * demand for (i, j), demand in required_flow.items())
        print(f"Required flow cost: {C_req}")

        used_nodes_list = list(used_nodes)
        flow_model = Model("Flow Optimization")
        flow_model.ModelSense = GRB.MINIMIZE

        # flow variables
        logging.info("Building flow variables")
        flow_vars = flow_model.addVars(
            used_nodes_list,
            used_nodes_list,
            vtype=GRB.INTEGER,
            name="phi",
            obj={(x, y): self.line_instance.dm[x][y] for x in used_nodes_list for y in used_nodes_list}
        )

        # required flow constraints
        for (i, j), demand in required_flow.items():
            if i == j:
                continue
            if (i, j) in flow_vars:
                flow_model.addConstr(flow_vars[(i, j)] >= demand, name=f"demand[{i},{j}]")

        # Flow conservation constraints
        logging.info("Building flow conservation constraints")
        flow_model.addConstrs(flow_vars.sum('*', j) - flow_vars.sum(j, '*') == 0 for j in used_nodes_list)

        flow_model.write(str(output_dir / "flow_ILP.lp"))

        flow_model.optimize()

        flow_model.write(str(output_dir / "flow_ILP.sol"))

        print("Model obj:", flow_model.ObjVal, " 2x baseline:", 2 * C_req)

        assert flow_model.ObjVal <= 2 * C_req + 1e-8, "Shouldn't exceed 2× under symmetric metric nonneg costs"

        flows = {
            (i, j): var.X
            for (i, j), var in flow_vars.items()
            if var.X > EPS
        }

        self._export_flows(flows, output_dir)

    def _export_flows(
        self,
        flows: Dict[Tuple[int, int], float],
        output_dir: Path
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"flows.csv"
        try:
            with csv_path.open("w", encoding="utf-8") as csv_file:
                csv_file.write("from,to,flow,cost\n")
                for (i, j), flow_value in sorted(flows.items()):
                    cost = self.line_instance.dm[i][j]
                    csv_file.write(f"{i},{j},{flow_value},{cost}\n")
            logging.info("Exported flows to %s", csv_path)
        except OSError as exc:
            logging.warning("Unable to write flows CSV %s: %s", csv_path, exc)

    # Implementation of the column generation process. Outputs the solution of the configuration LP before rounding.
    def solve_master_LP(self, gurobi_subproblem_method: Optional[int] = None):
        """
        Column-generation master LP for the proposed method.

        ``gurobi_subproblem_method``: optional Gurobi ``Method`` parameter for each
        single-line sub-MIP (omit or ``0`` for Gurobi default).
        """
        nb_pass = self.line_instance.nb_pass
        nb_lines = self.line_instance.nb_lines
        print(nb_pass, nb_lines)

        capacity = self.line_instance.capacity #capacity of a line

        t_1 = time.time()

        sets = [] #set[j] stores the indices of passengers present in the set of index j
        lines_to_sets = []	# contains, for each line l', the indices of the active sets of passengers
        for j in range(nb_lines * self.max_frequency):
            lines_to_sets.append([])

        # cost proportional to travel time on line
        lines_cost = [self.cost_coefficient * self.line_instance.lengths_travel_times[l//self.max_frequency] + l % self.max_frequency * self.line_instance.lengths_travel_times[l//self.max_frequency] for l in range(nb_lines * self.max_frequency)]

        passengers_to_sets = [[[] for l in range(nb_lines * self.max_frequency)] for p in range(nb_pass)] #for passenger index p and line l, contains the list of indices of sets including passenger p

        # generate initial sets with one passenger covered per line (if no passenger covered, add empty set)
        for l in range(len(lines_to_sets)):
            route = l // self.max_frequency
            p = self.line_instance.first_passenger_on_line(route)
            if p is not None:
                sets.append([p])
                lines_to_sets[l].append(l)
                passengers_to_sets[p][l].append(l)
            else:
                sets.append([])
                lines_to_sets[l].append(l)

        lines_to_passengers = []
        for l in range(len(lines_to_sets)):
            lines_to_passengers.append(self.line_instance.line_passengers(l // self.max_frequency))

        covered_average = 0
        iter = 0

        # Build master LP
        master = Model("LP")
        master.ModelSense = -1

        # Define variables
        x = {}
        for l in range(len(lines_to_sets)):
            for s in lines_to_sets[l]:
                total_set_value = sum(
                    [self.line_instance.trip_value_on_line(p, l // self.max_frequency) for p in sets[s]]
                )
                x[l,s] = master.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj = total_set_value, name="x[%d][%d]"%(l,s))
        master.update()

        one_set_per_passenger = {}

        for p in range(nb_pass):
            var = []
            for l in range(len(passengers_to_sets[p])):
                if len(passengers_to_sets[p][l]) > 0:
                    for s in passengers_to_sets[p][l]:
                        var.append(x[l,s])
            coef = [1 for i in range(len(var))]
            one_set_per_passenger[p] = master.addConstr(LinExpr(coef,var) <= 1, name="one_set_per_passenger[%d]"%p)
            master.update()

        master.update()

        master.Params.OutputFlag = 0 # silent mode

        one_set_per_line = {}
        for l in range(len(lines_to_sets)):
            var = [x[l,s] for s in lines_to_sets[l]]
            coef = [1 for i in range(len(var))]
            one_set_per_line[l] = master.addConstr(LinExpr(coef,var) <= 1, name="one_set_per_line[%d]"%l)

        var = []
        coef = []
        for l in range(len(lines_to_sets)):
            for s in lines_to_sets[l]:
                var.append(x[l,s])
                coef.append(lines_cost[l])

        print('Budget', self.line_instance.B)
        budget_constraint = master.addConstr(LinExpr(coef, var) <= self.line_instance.B, name="budget_constraint")
        master.update()

        # Define dual variables and constraints
        lamb = []
        alpha = 0
        q = []
        lamb_constr = []
        alpha_constr=None
        q_constr = []

        t_2 = time.time()

        print('card L', len(lines_to_sets))

        master.Params.timeLimit = self.time_limit

        master.optimize()

        t_4 = 0

        x_temp = {}
        for l in range(len(lines_to_sets)):
            for s in lines_to_sets[l]:
                x_temp[l,s] = deepcopy(x[l,s].X)

        obj_temp = master.ObjVal
        lines_to_sets_temp = deepcopy(lines_to_sets)

        while t_4 - t_2 <= self.time_limit:
            t_0 = time.time()
            print('iteration', iter)
            iter += 1

            # Retrieve values of dual variables
            lamb_constr = [master.getConstrByName("one_set_per_passenger[%d]"%p) for p in range(nb_pass)]
            lamb = [c.Pi for c in lamb_constr]
            q_constr = [master.getConstrByName("one_set_per_line[%d]"%l) for l in range(len(lines_to_sets))]
            q = [c.Pi for c in q_constr]
            alpha_constr = master.getConstrByName("budget_constraint") # keep dual variables
            alpha = alpha_constr.Pi

            found_a_new_set = False

            covered_average = 0
            nb_new_lines = 0
            max_nb_new_lines = 100 #new columns added in each iteration

            for l in range(len(lines_to_sets)):
                if nb_new_lines <= max_nb_new_lines:
                    t_temp = time.time()
                    f_l = l%self.max_frequency + 1
                    route = l // self.max_frequency
                    length = self.line_instance.line_length(route)

                    single_line = Model("SLP") #single line sub-problem
                    single_line.ModelSense = -1 #maximize

                    if gurobi_subproblem_method is not None and gurobi_subproblem_method != 0:
                        single_line.Params.OutputFlag = 0
                        single_line.Params.Method = gurobi_subproblem_method

                    y = {}
                    for p in lines_to_passengers[l]:
                        y[p] = single_line.addVar(obj=self.line_instance.trip_value_on_line(p, l // self.max_frequency) - lamb[p], ub=1, vtype=GRB.BINARY, name="y[%d]" % p)
                    single_line.update()
                    var = [y[p] for p in lines_to_passengers[l]]

                    for k in range(length):
                        coef = []
                        edge_ps = set(self.line_instance.edge_passengers(route, k))
                        for p in lines_to_passengers[l]:
                            coef.append(1 if p in edge_ps else 0)
                        single_line.addConstr(LinExpr(coef,var) <= capacity * f_l, name="one_set_per_line[%d]"%l)

                    single_line.update()
                    single_line.Params.OutputFlag = 0
                    single_line.optimize()
                    t_end = time.time()

                    if single_line.ObjVal >= q[l] + alpha * lines_cost[l] + EPS:

                        nb_new_lines +=1
                        found_a_new_set = True
                        new_set = [p for p in lines_to_passengers[l] if y[p].X > 0]
                        covered_average += len(new_set)
                        sets.append(new_set)
                        lines_to_sets[l].append(len(sets)-1)
                        for p in lines_to_passengers[l]:
                            if y[p].X > 0:
                                passengers_to_sets[p][l].append(len(sets)-1)

                        s = len(sets)-1

                        col = Column()
                        for p in new_set:
                            col.addTerms(1, one_set_per_passenger[p])
                        col.addTerms(1, one_set_per_line[l])
                        col.addTerms(lines_cost[l], budget_constraint)

                        total_set_value = sum(
                    [self.line_instance.trip_value_on_line(p, l // self.max_frequency) for p in sets[s]]
                )
                        x[l,s] = master.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj = total_set_value, name="x[%d][%d]"%(l,s), column = col)
                        t_update = time.time()

            if not found_a_new_set: #no more columns need to be added
                break
            master.update()
            t_3= time.time()

            if t_3 - t_2 <= self.time_limit:
                master.Params.timeLimit = self.time_limit - t_3 + t_2
                master.optimize()
                if master.ObjVal > obj_temp:
                    for l in range(len(lines_to_sets)):
                        for s in lines_to_sets[l]:
                            x_temp[l,s] = deepcopy(x[l,s].X)
                    obj_temp = master.ObjVal
                    lines_to_sets_temp = deepcopy(lines_to_sets)

            t_4 = time.time()

        print('---------------------------')
        t_fin = time.time()

        non_zero_var = 0
        budget = 0
        solution = {}
        active_sets = [[] for l in range(len(lines_to_sets_temp))]
        for l in range(len(lines_to_sets_temp)):
            l_activated = False
            for s in lines_to_sets_temp[l]:
                if x_temp[l,s]>0:
                    non_zero_var +=1
                    solution[l,s] = x_temp[l,s]
                    active_sets[l].append(s)
                    if not l_activated:
                        budget += lines_cost[l]
                    l_activated = True

        print('total_time', t_fin - t_2)
        print('number of non_zero_var: ', non_zero_var)
        print("final solution:  objective =", obj_temp)
        return solution, active_sets, sets, t_fin - t_2

    def solve_ILP(
        self,
        export_model: bool = False,
        export_solution: bool = False,
        output_dir: Union[Path, str, None] = None,
        gurobi_log_file: Union[Path, str, None] = None,
    ):
        request_count = self.line_instance.nb_pass
        line_count = self.line_instance.nb_lines
        bus_capacity = self.line_instance.capacity

        lines_cost = [
            self.cost_coefficient * self.line_instance.lengths_travel_times[l // self.max_frequency]
            + l
            % self.max_frequency
            * self.line_instance.lengths_travel_times[l // self.max_frequency]
            for l in range(line_count * self.max_frequency)
        ]

        master = Model("LP") # master LP problem
        master.ModelSense = -1 # maximize the objective function

        master.Params.timeLimit = self.time_limit

        # Line variables
        y = {} # binary variable indicating if line l is opened
        for l in range(self.line_count_total):
            y[l] = master.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.BINARY, obj = 0, name="y[%d]"%l)
        master.update()

        # Passenger variables
        x = {}
        for l in range(self.line_count_total):
            for p in range(request_count):
                val = self.line_instance.trip_value_on_line(p, l // self.max_frequency)
                if val > 0:
                    x[l,p] = master.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.BINARY, obj = val, name="x[%d][%d]"%(l,p))
        master.update()

        # One line per passenger constraints
        one_line_per_passenger = {}
        for p in range(request_count):
            var = []
            for l in range(self.line_count_total):
                val = self.line_instance.trip_value_on_line(p, l // self.max_frequency)
                if val > 0:
                    var.append(x[l,p])
            coef = [1 for j in range(len(var))]
            one_line_per_passenger[p] = master.addConstr(LinExpr(coef,var) <= 1, name="one_line_per_passenger[%d]"%p)
        master.update()

        # Bus capacity constraints
        capacity_constraints = {}
        for l in range(self.line_count_total):
            f_l = l%self.max_frequency + 1
            route = l // self.max_frequency
            length = self.line_instance.line_length(route)
            for k in range(length):
                var = []
                coef = []
                for p in self.line_instance.edge_passengers(route, k):
                    var.append(x[l,p])
                    coef.append(1)
                capacity_constraints[l,k] = master.addConstr(LinExpr(coef,var) <= bus_capacity * f_l * y[l], name="capacity_constraints[%d][%d]"%(l,k))
        master.update()

        # Budget constraint
        var = [y[l] for l in range(self.line_count_total)]
        coef = [lines_cost[l] for l in range(self.line_count_total)]
        budget_constraint = master.addConstr(LinExpr(coef,var) <= self.line_instance.B, name="budget_constraints")
        master.update()

        output_dir_path: Optional[Path] = Path(output_dir) if output_dir is not None else None
        if gurobi_log_file is not None:
            gurobi_log_path = Path(gurobi_log_file)
            gurobi_log_path.parent.mkdir(parents=True, exist_ok=True)
            master.Params.LogFile = str(gurobi_log_path)

        logging.info('method: %s', master.Params.Method)
        t0 = time.time()
        master.optimize()
        t1 = time.time()
        logging.info("Execution time: %s", t1-t0)
        logging.info("Final solution: %s", master.ObjVal)

        if export_model and output_dir_path is not None:
            output_dir_path.mkdir(parents=True, exist_ok=True)
            master.write(str(output_dir_path / "ILP.lp"))

        if export_solution and output_dir_path is not None:
            output_dir_path.mkdir(parents=True, exist_ok=True)
            master.write(str(output_dir_path / "ILP.sol"))

        export_dir = output_dir_path if output_dir_path is not None else Path(".")
        self._export_used_lines(
            output_dir=export_dir,
            line_vars=y,
            line_costs=lines_cost
        )

        partial_assignments = self._assignments_from_passenger_vars(
            passenger_vars=x,
        )
        _request_assignments, assignments = self._resolve_and_export_request_assignments(
            output_dir=export_dir,
            request_assignments=partial_assignments,
            no_assignment_handling=NoAssignmentHandling.NO_MT,
        )

        self._solve_and_export_flows(
            assignments=assignments,
            output_dir=export_dir
        )

        return master.ObjVal, t1-t0

    
    def solve_modified_ILP(
        self,
        export_model: bool = False,
        export_solution: bool = False,
        output_dir: Union[Path, str] = Path("."),
        gurobi_log_file: Union[Path, str, None] = None,
    ):
        """Solve the problem as described in Section 3 of stage_1_formulation.pdf. It is 
        a modification of (Périvier et al., 2021), where the MoD is also considered under the same budget constraint
        as the lines.

        Args:
            export_model (bool, optional): Defaults to False.
            export_solution (bool, optional): Defaults to False.
            output_dir (Union[Path, str], optional): Defaults to the current directory.
            gurobi_log_file (Union[Path, str, None], optional): Defaults to None.

        Returns:
            :class:`LineSelectionSolveResult` (also iterable as ``(obj, time, lines, assignments, line_c, mod_c)``).
        """

        request_count = self.line_instance.nb_pass
        bus_capacity = self.line_instance.capacity
        nb_lines = self.line_instance.nb_lines
        freq_ub = self.max_frequency
        no_mt_key = nb_lines

        master = Model("Modified ILP")  # master LP problem
        master.ModelSense = -1  # maximize the objective function

        master.Params.timeLimit = self.time_limit

        # Integer frequency y_ρ per route ρ (same aggregation as solve_MoD_aware_ILP §4.1.1).
        frequency_vars = master.addVars(
            nb_lines,
            vtype=GRB.INTEGER,
            lb=0,
            ub=freq_ub,
            name="y",
        )

        per_route_mt_cost_coeff = [
            self.cost_coefficient * self.line_instance.lengths_travel_times[rho]
            for rho in range(nb_lines)
        ]
        line_costs_expression = frequency_vars.prod(per_route_mt_cost_coeff)

        # Binary assignment x_{ρp}; omit pairs with zero trip value on the route.
        logging.info("Computing potential line-passenger combinations")
        potential_line_passenger_combinations = self.line_instance.positive_trip_value_pairs()
        passenger_vars = master.addVars(
            potential_line_passenger_combinations,
            vtype=GRB.BINARY,
            obj=1,
            name="x",
        )
        for p in range(request_count):
            passenger_vars[no_mt_key, p] = master.addVar(
                vtype=GRB.BINARY,
                obj=1,
                name="x[no_MT,%d]" % p,
            )

        # One line per passenger constraints
        master.addConstrs(
            (passenger_vars.sum("*", p) <= 1 for p in tqdm(range(request_count), desc="Adding one line per passenger constraints")),
            name="one_line_per_passenger",
        )

        # Bus capacity: load on edge ≤ C_MT · y_ρ
        for rho in range(nb_lines):
            length = self.line_instance.line_length(rho)
            for k in range(length):
                vars_list = []
                coefs_list = []
                for p in self.line_instance.edge_passengers(rho, k):
                    if (rho, p) in passenger_vars:
                        vars_list.append(passenger_vars[rho, p])
                        coefs_list.append(1)
                if vars_list:
                    master.addConstr(
                        LinExpr(coefs_list, vars_list) <= bus_capacity * frequency_vars[rho],
                        name="capacity_constraints[%d][%d]" % (rho, k),
                    )

        # Budget constraint
        mod_costs: Dict[Any, float] = {}
        for rho, p in potential_line_passenger_combinations:
            mod_costs[rho, p] = self.line_instance.trip_mod_cost_on_line(p, rho)
        for p in range(request_count):
            mod_costs[no_mt_key, p] = float(self._direct_trip_mod_cost(p))
        mod_cost_expression = passenger_vars.prod(mod_costs)
        master.addConstr(
            line_costs_expression + mod_cost_expression <= self.line_instance.B,
            name="budget_constraint",
        )

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        if gurobi_log_file is not None:
            gurobi_log_path = Path(gurobi_log_file)
            gurobi_log_path.parent.mkdir(parents=True, exist_ok=True)
            master.Params.LogFile = str(gurobi_log_path)

        if export_model:
            logging.info("Exporting model to %s", output_dir_path / "ILP.lp")
            master.write(str(output_dir_path / "ILP.lp"))

        t0 = time.time()
        master.optimize()
        t1 = time.time()

        logging.info("Execution time: %s", t1-t0)
        logging.info("Final solution cost: %s", master.ObjVal)

        if export_solution:
            logging.info("Exporting solution to %s", output_dir_path / "ILP.sol")
            master.write(str(output_dir_path / "ILP.sol"))

        self._export_used_lines_route_agg(
            output_dir=output_dir_path,
            frequency_vars=frequency_vars,
            per_route_mt_cost_coeff=per_route_mt_cost_coeff,
        )

        partial_assignments = self._assignments_from_passenger_vars(
            passenger_vars=passenger_vars,
            no_mt_line_key=no_mt_key,
            line_var_is_route_index=True,
        )
        request_assignments_list, assignments = self._resolve_and_export_request_assignments(
            output_dir=output_dir_path,
            request_assignments=partial_assignments,
            no_assignment_handling=NoAssignmentHandling.REJECT,
            no_mt_line_key=no_mt_key,
        )

        # self._solve_and_export_flows(
        #     assignments=assignments,
        #     output_dir=output_dir_path,
        # )

        selected_lines: List[int] = [
            rho for rho in range(nb_lines) if frequency_vars[rho].X > EPS
        ]

        line_obj_val: Optional[float] = None
        mod_obj_val: Optional[float] = None
        try:
            line_obj_val = float(line_costs_expression.getValue())
            mod_obj_val = float(mod_cost_expression.getValue())
        except (GurobiError, AttributeError, TypeError, ValueError):
            pass

        try:
            obj_v = float(master.ObjVal)
        except (GurobiError, TypeError, ValueError):
            obj_v = float("nan")

        return LineSelectionSolveResult(
            objective_value=obj_v,
            run_time_seconds=float(t1 - t0),
            selected_lines=selected_lines,
            request_assignments=tuple(request_assignments_list),
            line_objective_component=line_obj_val,
            mod_objective_component=mod_obj_val,
        )

    def _dispose_mod_aware_mip_state(self) -> None:
        if self._mod_aware_mip_state is None:
            return
        try:
            self._mod_aware_mip_state["model"].dispose()
        except Exception as exc:
            logging.warning("Error disposing MoD-aware MIP model: %s", exc)
        self._mod_aware_mip_state = None

    @staticmethod
    def _mod_aware_ilp_fingerprint(
        freq_ub: int,
        allow_rejection: bool,
        use_request_line_valid_inequalities: bool,
        nb_lines: int,
        request_count: int,
        bus_capacity: int,
    ) -> Tuple[Any, ...]:
        return (
            freq_ub,
            allow_rejection,
            use_request_line_valid_inequalities,
            nb_lines,
            request_count,
            bus_capacity,
        )

    def _update_mod_aware_mip_objective(self, state: Dict[str, Any]) -> None:
        """Re-set objective with fresh passenger MoD coefficients; constraints unchanged."""
        master: Model = state["model"]
        passenger_vars = state["passenger_vars"]
        no_mt_key: int = state["no_mt_key"]
        request_count: int = state["request_count"]
        mod_costs_for_obj: Dict[Any, float] = {}
        for rho, p in state["potential_line_passenger_combinations"]:
            mod_costs_for_obj[(rho, p)] = self.line_instance.trip_mod_cost_on_line(p, rho)
        for p in range(request_count):
            mod_costs_for_obj[no_mt_key, p] = float(self._direct_trip_mod_cost(p))
        mod_cost_expression = passenger_vars.prod(mod_costs_for_obj)
        if state["allow_rejection"]:
            master.setObjective(
                state["line_costs_expression"] + mod_cost_expression + state["rejection_expr"],
                GRB.MINIMIZE,
            )
        else:
            master.setObjective(state["line_costs_expression"] + mod_cost_expression, GRB.MINIMIZE)
        master.update()

    def _create_mod_aware_mip_state(
        self,
        fingerprint: Tuple[Any, ...],
        freq_ub: int,
        rejection_cost: float,
        use_request_line_valid_inequalities: bool,
    ) -> Dict[str, Any]:

        logging.info("Building MoD-aware ILP model")

        nb_lines = self.line_instance.nb_lines
        request_count = self.line_instance.nb_pass
        bus_capacity = self.line_instance.capacity
        no_mt_key = nb_lines
        rej_penalty = float(rejection_cost) if rejection_cost is not None else 0.0
        allow_rejection = rej_penalty > 0.0

        master = Model("MoD-aware ILP")
        master.ModelSense = GRB.MINIMIZE

        master.Params.timeLimit = self.time_limit

        per_route_mt_cost_coeff = [
            self.cost_coefficient * self.line_instance.lengths_travel_times[rho] for rho in range(nb_lines)
        ]

        frequency_vars = master.addVars(
            nb_lines,
            vtype=GRB.INTEGER,
            lb=0,
            ub=freq_ub,
            name="y",
        )

        potential_line_passenger_combinations = self.line_instance.positive_trip_value_pairs()

        logging.info("Computing total mod costs for each trip option")
        mod_costs_line = {
            (rho, p): self.line_instance.trip_mod_cost_on_line(p, rho)
            for (rho, p) in tqdm(potential_line_passenger_combinations, desc="Computing trip option mod costs")
        }
        passenger_vars = master.addVars(
            potential_line_passenger_combinations,
            vtype=GRB.BINARY,
            obj=0,
            name="x",
        )
        for p in range(request_count):
            passenger_vars[no_mt_key, p] = master.addVar(
                vtype=GRB.BINARY,
                obj=0,
                name="x[no_MT,%d]" % p,
            )
        master.update()

        line_costs_expression = frequency_vars.prod(per_route_mt_cost_coeff)
        mod_costs_for_obj = dict(mod_costs_line)

        logging.info("Adding direct trip mod costs to the objective")
        for p in tqdm(range(request_count), desc="Adding direct trip for request"):
            mod_costs_for_obj[no_mt_key, p] = self._direct_trip_mod_cost(p)
        mod_cost_expression = passenger_vars.prod(mod_costs_for_obj)
        rej_vars = None
        rejection_expr = None
        if allow_rejection:
            rej_vars = master.addVars(request_count, vtype=GRB.BINARY, name="xrej")
            rejection_expr = rej_penalty * quicksum(rej_vars[p] for p in range(request_count))
            master.setObjective(line_costs_expression + mod_cost_expression + rejection_expr, GRB.MINIMIZE)
        else:
            master.setObjective(line_costs_expression + mod_cost_expression, GRB.MINIMIZE)

        if allow_rejection:
            assert rej_vars is not None
            master.addConstrs(
                (passenger_vars.sum("*", p) + rej_vars[p] == 1 for p in range(request_count)),
                name="one_option_or_reject_per_passenger",
            )
        else:
            master.addConstrs(
                (passenger_vars.sum("*", p) == 1 for p in range(request_count)),
                name="one_option_per_passenger",
            )

        for rho in range(nb_lines):
            length = self.line_instance.line_length(rho)
            for k in range(length):
                vars_list = []
                coefs_list = []
                for p in self.line_instance.edge_passengers(rho, k):
                    if (rho, p) in passenger_vars:
                        vars_list.append(passenger_vars[rho, p])
                        coefs_list.append(1)
                if vars_list:
                    master.addConstr(
                        LinExpr(coefs_list, vars_list) <= bus_capacity * frequency_vars[rho],
                        name="capacity[%d][%d]" % (rho, k),
                    )
        if use_request_line_valid_inequalities:
            logging.info("Adding request-line valid inequalities")
            for rho, p in tqdm(potential_line_passenger_combinations, desc="Adding request-line valid inequalities"):
                master.addConstr(
                    passenger_vars[rho, p] <= frequency_vars[rho],
                    name=f"x_le_y[{rho},{p}]",
                )
            # master.addConstrs(
            #     (
            #         passenger_vars[rho, p] <= frequency_vars[rho]
            #         for rho, p in potential_line_passenger_combinations
            #     ),
            #     name="x_le_y",
            # )
        master.update()

        return {
            "fingerprint": fingerprint,
            "model": master,
            "frequency_vars": frequency_vars,
            "passenger_vars": passenger_vars,
            "rej_vars": rej_vars,
            "allow_rejection": allow_rejection,
            "rejection_expr": rejection_expr,
            "rej_penalty": rej_penalty,
            "no_mt_key": no_mt_key,
            "nb_lines": nb_lines,
            "request_count": request_count,
            "per_route_mt_cost_coeff": per_route_mt_cost_coeff,
            "line_costs_expression": line_costs_expression,
            "potential_line_passenger_combinations": potential_line_passenger_combinations,
        }

    def _peak_batch_request_indices(self, max_wait_time: float) -> Tuple[List[int], int]:
        beta = float(max_wait_time)
        if beta <= 0 or not np.isfinite(beta):
            raise ValueError(f"max_wait_time must be a positive finite number, got {max_wait_time!r}")

        demand_df = self.line_instance.demand
        request_count = self.line_instance.nb_pass
        if "time" not in demand_df.columns:
            raise ValueError("line_instance.demand must contain a 'time' column")
        if len(demand_df) != request_count:
            raise ValueError(
                f"line_instance.demand has {len(demand_df)} rows, expected nb_pass={request_count}"
            )

        demand_times = demand_df["time"].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(demand_times).all():
            raise ValueError("line_instance.demand['time'] contains NaN or infinite values")
        batch_ids = np.floor(demand_times / beta).astype(np.int64, copy=False)
        unique_batches, counts = np.unique(batch_ids, return_counts=True)
        if unique_batches.size == 0:
            return [], 0
        peak_batch_id = int(unique_batches[int(np.argmax(counts))])
        peak_indices = np.flatnonzero(batch_ids == peak_batch_id).astype(np.int32, copy=False)
        return [int(p) for p in peak_indices], peak_batch_id

    def _assign_remaining_requests_to_selected_routes(
        self,
        selected_routes: List[int],
        fixed_assignments: Dict[int, Tuple[str, Optional[int]]],
    ) -> List[Tuple[str, Optional[int]]]:
        """
        Build full-demand assignments after a peak-batch solve.

        Requests fixed by the ILP keep their assignment. Other requests choose the cheapest
        MoD-cost option among direct MoD and feasible options on selected routes. Equal costs
        keep the earlier option, so direct MoD wins ties and route ties go to the lower index.
        """
        selected_routes_sorted = sorted(int(rho) for rho in selected_routes)
        request_assignments: List[Tuple[str, Optional[int]]] = []
        for p in range(self.line_instance.nb_pass):
            fixed = fixed_assignments.get(p)
            if fixed is not None:
                request_assignments.append(fixed)
                continue

            best_assignment: Tuple[str, Optional[int]] = ("no_MT", None)
            best_cost = float(self._direct_trip_mod_cost(p))
            for rho in selected_routes_sorted:
                if not self.line_instance.has_trip_option_on_line(p, rho):
                    continue
                route_cost = float(self.line_instance.trip_mod_cost_on_line(p, rho))
                if route_cost < best_cost - EPS:
                    best_cost = route_cost
                    best_assignment = ("line", rho)
            request_assignments.append(best_assignment)
        return request_assignments

    def solve_peak_batch_max_headway_ILP(
        self,
        export_model: bool = False,
        export_solution: bool = False,
        output_dir: Union[Path, str] = Path("."),
        gurobi_log_file: Union[Path, str, None] = None,
        max_route_frequency: Optional[int] = None,
        max_wait_time: float = 300.0,
    ) -> LineSelectionSolveResult:
        """
        Solve the peak-batch max-headway route-aggregated ILP from manuscript §4.2.1.

        The ILP is built only over the largest request-time batch where the batch width is
        ``max_wait_time``. After solving, requests outside the peak batch are assigned to
        direct MoD or the cheapest feasible selected route by MoD cost, so downstream DARP
        evaluation still receives one assignment per original request.
        """
        freq_ub = max_route_frequency if max_route_frequency is not None else self.max_frequency
        max_wait_time = float(max_wait_time)

        nb_lines = self.line_instance.nb_lines
        request_count = self.line_instance.nb_pass
        bus_capacity = self.line_instance.capacity
        no_mt_key = nb_lines
        peak_requests, peak_batch_id = self._peak_batch_request_indices(max_wait_time)
        peak_request_set = set(peak_requests)

        logging.info(
            "Building peak-batch max-headway ILP (batch_id=%s, peak_size=%s, max_wait_time=%s)",
            peak_batch_id,
            len(peak_requests),
            max_wait_time,
        )

        master = Model("Peak-batch max-headway ILP")
        master.ModelSense = GRB.MINIMIZE
        master.Params.timeLimit = self.time_limit

        per_route_mt_cost_coeff = [
            self.cost_coefficient * self.line_instance.lengths_travel_times[rho] for rho in range(nb_lines)
        ]
        frequency_vars = master.addVars(
            nb_lines,
            vtype=GRB.INTEGER,
            lb=0,
            ub=freq_ub,
            name="y",
        )

        potential_line_passenger_combinations = [
            (rho, p)
            for (rho, p) in self.line_instance.positive_trip_value_pairs()
            if p in peak_request_set
        ]
        passenger_vars = master.addVars(
            potential_line_passenger_combinations,
            vtype=GRB.BINARY,
            obj=0,
            name="x",
        )
        for p in peak_requests:
            passenger_vars[no_mt_key, p] = master.addVar(
                vtype=GRB.BINARY,
                obj=0,
                name="x[no_MT,%d]" % p,
            )
        master.update()

        line_costs_expression = frequency_vars.prod(per_route_mt_cost_coeff)
        mod_costs_for_obj: Dict[Any, float] = {
            (rho, p): self.line_instance.trip_mod_cost_on_line(p, rho)
            for (rho, p) in potential_line_passenger_combinations
        }
        for p in peak_requests:
            mod_costs_for_obj[no_mt_key, p] = float(self._direct_trip_mod_cost(p))
        mod_cost_expression = passenger_vars.prod(mod_costs_for_obj)
        master.setObjective(line_costs_expression + mod_cost_expression, GRB.MINIMIZE)

        master.addConstrs(
            (passenger_vars.sum("*", p) == 1 for p in peak_requests),
            name="one_option_per_peak_passenger",
        )

        route_travel_times = [
            float(self.line_instance.lengths_travel_times[rho]) for rho in range(nb_lines)
        ]
        for rho, p in potential_line_passenger_combinations:
            nu_rho = route_travel_times[rho] / max_wait_time
            master.addConstr(
                frequency_vars[rho] >= nu_rho * passenger_vars[rho, p],
                name=f"max_headway[{rho},{p}]",
            )

        for rho in range(nb_lines):
            length = self.line_instance.line_length(rho)
            for k in range(length):
                vars_list = []
                coefs_list = []
                for p in self.line_instance.edge_passengers(rho, k):
                    if p in peak_request_set and (rho, int(p)) in passenger_vars:
                        vars_list.append(passenger_vars[rho, int(p)])
                        coefs_list.append(1)
                if vars_list:
                    master.addConstr(
                        LinExpr(coefs_list, vars_list) <= bus_capacity * frequency_vars[rho],
                        name="capacity_peak[%d][%d]" % (rho, k),
                    )
        master.update()

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        if gurobi_log_file is not None:
            gurobi_log_path = Path(gurobi_log_file)
            gurobi_log_path.parent.mkdir(parents=True, exist_ok=True)
            master.Params.LogFile = str(gurobi_log_path)

        if export_model:
            master.write(str(output_dir_path / "Peak_batch_max_headway_ILP.lp"))

        t0 = time.time()
        master.optimize()
        t1 = time.time()

        logging.info("Execution time: %s", t1 - t0)
        logging.info("Final solution (peak-batch total cost): %s", master.ObjVal)

        if export_solution:
            master.write(str(output_dir_path / "Peak_batch_max_headway_ILP.sol"))

        selected_lines = [
            rho for rho in range(nb_lines) if frequency_vars[rho].X > EPS
        ]

        fixed_assignments: Dict[int, Tuple[str, Optional[int]]] = {}
        for p in peak_requests:
            if passenger_vars[no_mt_key, p].X > EPS:
                fixed_assignments[p] = ("no_MT", None)
                continue
            assigned_route: Optional[int] = None
            for rho in range(nb_lines):
                if (rho, p) in passenger_vars and passenger_vars[rho, p].X > EPS:
                    assigned_route = rho
                    break
            fixed_assignments[p] = ("line", assigned_route) if assigned_route is not None else ("no_MT", None)

        request_assignments_list = self._assign_remaining_requests_to_selected_routes(
            selected_lines,
            fixed_assignments,
        )

        self._export_used_lines_route_agg(
            output_dir=output_dir_path,
            frequency_vars=frequency_vars,
            per_route_mt_cost_coeff=per_route_mt_cost_coeff,
        )
        request_assignments_list, _assignments = self._resolve_and_export_request_assignments(
            output_dir=output_dir_path,
            request_assignments=request_assignments_list,
            no_assignment_handling=NoAssignmentHandling.RAISE,
            no_mt_line_key=no_mt_key,
        )

        peak_info = {
            "peak_batch_id": peak_batch_id,
            "peak_batch_size": len(peak_requests),
            "max_wait_time": max_wait_time,
            "batch_request_indices": peak_requests,
        }
        try:
            (output_dir_path / "peak_batch_info.json").write_text(json.dumps(peak_info, indent=2))
        except OSError as exc:
            logging.warning("Unable to write peak batch info JSON: %s", exc)

        line_obj_val: Optional[float] = None
        mod_obj_val: Optional[float] = None
        try:
            line_obj_val = float(line_costs_expression.getValue())
            mod_obj_val = float(mod_cost_expression.getValue())
            obj_v = float(master.ObjVal)
        except (GurobiError, AttributeError, TypeError, ValueError):
            obj_v = float("nan")

        return LineSelectionSolveResult(
            objective_value=obj_v,
            run_time_seconds=float(t1 - t0),
            selected_lines=selected_lines,
            request_assignments=tuple(request_assignments_list),
            line_objective_component=line_obj_val,
            mod_objective_component=mod_obj_val,
        )

    def solve_MoD_aware_ILP(
        self,
        export_model: bool = False,
        export_solution: bool = False,
        output_dir: Union[Path, str] = Path("."),
        gurobi_log_file: Union[Path, str, None] = None,
        max_route_frequency: Optional[int] = None,
        rejection_cost: float = 0.0,
        use_request_line_valid_inequalities: bool = False,
        reuse_model: bool = False,
    ):
        """
        Solve the MoD-aware line selection ILP (manuscript §4.1) using the route-aggregated
        formulation from §4.1.1: integer frequency y_ρ per route ρ, binary assignment x_ρr.

        Objective: min sum_r f̃tc(τ^MoD_r) x^MoD_r + sum_{ρ,r} f̃tc(τ*_{ρr}) x_{ρr}
                    + sum_ρ (∑_{e∈ρ} c_e) y_ρ
        with ∑_{e∈ρ} c_e approximated by cost_coefficient · lengths_travel_times[ρ].

        When ``rejection_cost`` > 0, §4.1.2 applies: binary x^rej_r per request, objective term
        ``rejection_cost`` · x^rej_r, and x^MoD_r + sum_ρ x_{ρr} + x^rej_r = 1 for each request.
        For ``rejection_cost`` == 0 (default), rejection variables are omitted (all requests served).

        If ``use_request_line_valid_inequalities`` is True, add valid linking inequalities
        x_{ρr} ≤ y_ρ for every route ρ and request r with a line-assignment variable (strengthens
        the LP relaxation; optional).

        Constraints: one option per request; edge capacity sum_{r: e∈ρ_{ρr}} x_{ρr} ≤ C_MT y_ρ;
        0 ≤ y_ρ ≤ max_route_frequency (when omitted, the solver's route-frequency cap applies).

        If ``reuse_model`` is True, keep one Gurobi model on this solver and only refresh the
        passenger MoD terms in the objective between solves. The previous solution stays feasible
        when only objective coefficients change; we do not call ``Model.reset()`` or set
        parameters that discard MIP starts (e.g. ``IgnoreStart``).

        Returns:
            :class:`LineSelectionSolveResult` with objective value, wall-clock seconds, selected route
            indices, request assignments, line objective term (sum_ρ cost_coefficient·length·y_ρ), and
            MoD objective term (passenger mod costs). The last two fields are ``None`` if the objective
            could not be decomposed from the solution.

            The result is iterable as a 6-tuple for backward compatibility.
        """
        freq_ub = max_route_frequency if max_route_frequency is not None else self.max_frequency

        nb_lines = self.line_instance.nb_lines
        request_count = self.line_instance.nb_pass
        bus_capacity = self.line_instance.capacity
        no_mt_key = nb_lines
        rej_penalty = float(rejection_cost) if rejection_cost is not None else 0.0
        allow_rejection = rej_penalty > 0.0

        fingerprint = self._mod_aware_ilp_fingerprint(
            freq_ub,
            allow_rejection,
            use_request_line_valid_inequalities,
            nb_lines,
            request_count,
            bus_capacity,
        )

        if not reuse_model:
            logging.info("Disposing old MoD-aware ILP model")
            self._dispose_mod_aware_mip_state()

        if reuse_model and self._mod_aware_mip_state is not None:
            if self._mod_aware_mip_state["fingerprint"] != fingerprint:
                logging.info("MoD-aware ILP fingerprint changed; rebuilding Gurobi model")
                self._dispose_mod_aware_mip_state()

        if self._mod_aware_mip_state is None:
            self._mod_aware_mip_state = self._create_mod_aware_mip_state(
                fingerprint,
                freq_ub,
                rejection_cost,
                use_request_line_valid_inequalities,
            )
            logging.info("Built MoD-aware ILP model (reuse_model=%s)", reuse_model)
        else:
            self._update_mod_aware_mip_objective(self._mod_aware_mip_state)
            logging.info(
                "Reusing MoD-aware ILP model; updated passenger MoD objective coefficients only",
            )

        state = self._mod_aware_mip_state
        master: Model = state["model"]
        frequency_vars = state["frequency_vars"]
        passenger_vars = state["passenger_vars"]
        rej_vars = state["rej_vars"]
        allow_rejection = state["allow_rejection"]
        rejection_expr = state["rejection_expr"]
        per_route_mt_cost_coeff = state["per_route_mt_cost_coeff"]
        line_costs_expression = state["line_costs_expression"]

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        if gurobi_log_file is not None:
            gurobi_log_path = Path(gurobi_log_file)
            gurobi_log_path.parent.mkdir(parents=True, exist_ok=True)
            master.Params.LogFile = str(gurobi_log_path)

        if export_model:
            master.write(str(output_dir_path / "MoD_aware_ILP.lp"))

        # master.Params.Method = 1 # use the dual simplex method for LP relaxation
        # master.Params.Method = 0 # use the primal simplex method for LP relaxation

        logging.info(
            "Solving MoD-aware ILP (section 4.1, route-aggregated §4.1.1, "
            "use_request_line_valid_inequalities=%s)",
            use_request_line_valid_inequalities,
        )
        t0 = time.time()
        master.optimize()
        t1 = time.time()

        logging.info("Execution time: %s", t1 - t0)
        logging.info("Final solution (total cost): %s", master.ObjVal)

        line_obj_val = None
        mod_obj_val = None
        try:
            line_obj_val = float(line_costs_expression.getValue())
            mod_costs_post: Dict[Any, float] = {}
            for rho, p in state["potential_line_passenger_combinations"]:
                mod_costs_post[(rho, p)] = self.line_instance.trip_mod_cost_on_line(p, rho)
            for p in range(request_count):
                mod_costs_post[no_mt_key, p] = float(self._direct_trip_mod_cost(p))
            mod_obj_val = float(
                sum(
                    float(passenger_vars[k].X) * mod_costs_post[k]
                    for k in mod_costs_post
                )
            )
            logging.info(
                "MoD-aware ILP objective breakdown: line=%s, mod=%s",
                line_obj_val,
                mod_obj_val,
            )
        except GurobiError:
            pass

        if allow_rejection and rejection_expr is not None:
            try:
                rej_val = float(rejection_expr.getValue())
                logging.info("MoD-aware ILP rejection-cost component: %s", rej_val)
            except GurobiError:
                pass

        if export_solution:
            master.write(str(output_dir_path / "MoD_aware_ILP.sol"))

        self._export_used_lines_route_agg(
            output_dir=output_dir_path,
            frequency_vars=frequency_vars,
            per_route_mt_cost_coeff=per_route_mt_cost_coeff,
        )

        rej_vars_map = {p: rej_vars[p] for p in range(request_count)} if rej_vars is not None else None
        partial_assignments = self._assignments_from_passenger_vars(
            passenger_vars=passenger_vars,
            no_mt_line_key=no_mt_key,
            line_var_is_route_index=True,
            rejection_vars=rej_vars_map,
        )
        request_assignments_list, assignments = self._resolve_and_export_request_assignments(
            output_dir=output_dir_path,
            request_assignments=partial_assignments,
            no_assignment_handling=NoAssignmentHandling.REJECT
            if allow_rejection
            else NoAssignmentHandling.RAISE,
            no_mt_line_key=no_mt_key,
        )

        # self._solve_and_export_flows(
        #     assignments=assignments,
        #     output_dir=output_dir_path,
        # )

        selected_lines = [
            rho for rho in range(nb_lines) if frequency_vars[rho].X > EPS
        ]
        try:
            obj_v = float(master.ObjVal)
        except (GurobiError, TypeError, ValueError):
            obj_v = float("nan")

        return LineSelectionSolveResult(
            objective_value=obj_v,
            run_time_seconds=float(t1 - t0),
            selected_lines=selected_lines,
            request_assignments=tuple(request_assignments_list),
            line_objective_component=line_obj_val,
            mod_objective_component=mod_obj_val,
        )

    def solve_ILP_with_empty_trips(
        self,
        export_model: bool = False,
        export_solution: bool = False,
        output_dir: Union[Path, str] = Path("."),
        gurobi_log_file: Union[Path, str, None] = None,
    ):
        request_count = self.line_instance.nb_pass
        bus_capacity = self.line_instance.capacity

        logging.info("Building ILP model with empty trips")

        master = Model("ILP with empty trips") # master LP problem
        master.ModelSense = -1 # maximize the objective function

        master.Params.timeLimit = self.time_limit

        # VARIABLES
        # binary variables indicating if line l is opened
        line_vars = master.addVars(self.line_count_total, vtype=GRB.BINARY, name="y")

        # binary variables indicating if passenger p is assigned to line l. If first mile + last mile costs are
        # higher than the no_MT MoD cost, the line-passenger combination is not considered at all
        route_passenger_pairs = self.line_instance.positive_trip_value_pairs()
        potential_line_passenger_combinations = [
            (rho * self.max_frequency + f, p)
            for rho, p in route_passenger_pairs
            for f in range(self.max_frequency)
        ]
        passenger_vars = master.addVars(potential_line_passenger_combinations, vtype=GRB.BINARY, obj=1, name="x")
        # add no MT variables for each passenger
        for p in range(request_count):
            passenger_vars[self.line_count_total, p] = master.addVar(vtype=GRB.BINARY, obj=1, name="x[no_MT,%d]" % p)

        # collections for the MoD flow constraints
        first_mile_vars = {}
        last_mile_vars = {}
        no_mt_vars = {}

        
        # first iterate over all request/line combinations and
        # 1. compute used nodes
        # 2. compute first mile vars, last mile vars, and no MT vars for the MoD flow constraints
        used_nodes = set()
        for p in tqdm(range(request_count), desc="Processing requests (used nodes, MoD flow constraints data...)"):
            request_from = self.line_instance.requests[p][0]
            request_to = self.line_instance.requests[p][1]
            used_nodes.add(request_from)
            used_nodes.add(request_to)
            for l in range(self.line_count_total):
                if (l, p) in passenger_vars:
                    mt_pickup_node, mt_drop_off_node = self.line_instance.trip_pickup_dropoff_on_line(
                        p,
                        l // self.max_frequency,
                    )
                    used_nodes.add(mt_pickup_node)
                    used_nodes.add(mt_drop_off_node)

                    # ALSO DO THE HARD WORK HERE FOR THE MOD FLOW CONSTRAINTS
                    # first mile vars
                    if request_from not in first_mile_vars:
                        first_mile_vars[request_from] = {}
                    if mt_pickup_node not in first_mile_vars[request_from]:
                        first_mile_vars[request_from][mt_pickup_node] = []
                    first_mile_vars[request_from][mt_pickup_node].append(passenger_vars[l,p])
                    # last mile vars
                    if mt_drop_off_node not in last_mile_vars:
                        last_mile_vars[mt_drop_off_node] = {}
                    if request_to not in last_mile_vars[mt_drop_off_node]:
                        last_mile_vars[mt_drop_off_node][request_to] = []
                    last_mile_vars[mt_drop_off_node][request_to].append(passenger_vars[l,p])
            # no MT vars
            if request_from not in no_mt_vars:
                no_mt_vars[request_from] = {}
            if request_to not in no_mt_vars[request_from]:
                no_mt_vars[request_from][request_to] = []
            no_mt_vars[request_from][request_to].append(passenger_vars[self.line_count_total,p])
        used_nodes_list = list(used_nodes)

        # integer flow variables indicating the MoD flow on each edge of the complete graph
        logging.info("Building flow variables")
        flow_vars = master.addVars(used_nodes_list, used_nodes_list, vtype=GRB.INTEGER, name="phi")

        # One line per passenger constraints
        logging.info("Building one line per passenger constraints")
        master.addConstrs(
            (passenger_vars.sum('*', p) <= 1 for p in range(request_count)),
            name="one_line_per_passenger"
        )

        # Bus capacity constraints
        logging.info("Building capacity constraints")
        capacity_constraints = {}
        for l in range(self.line_count_total):
            f_l = l%self.max_frequency + 1
            route = l // self.max_frequency
            length = self.line_instance.line_length(route)
            for k in range(length):
                vars = []
                coefs = []
                for p in self.line_instance.edge_passengers(route, k):
                    vars.append(passenger_vars[l,p])
                    coefs.append(1)
                capacity_constraints[l,k] = master.addConstr(LinExpr(coefs,vars) <= bus_capacity * f_l * line_vars[l], name="capacity_constraints[%d][%d]"%(l,k))

        # Budget constraint
        logging.info("Building budget constraint")
        line_costs = [
            self.cost_coefficient * self.line_instance.lengths_travel_times[l // self.max_frequency]
            + l
            % self.max_frequency
            * self.line_instance.lengths_travel_times[l // self.max_frequency]
            for l in range(self.line_count_total)
        ]
        line_costs_expression = line_vars.prod(line_costs)
        flow_costs = {}
        used_nodes_count = len(used_nodes_list)
        for flow_from in used_nodes:
            for flow_to in used_nodes:
                flow_costs[flow_from,flow_to] = self.line_instance.dm[flow_from][flow_to] # currently, distance is equal to the MoD cost
        flow_cost_expression = flow_vars.prod(flow_costs)
        master.addConstr(line_costs_expression + flow_cost_expression <= self.line_instance.B, name="budget_constraint")

        # Flow conservation constraints
        logging.info("Building flow conservation constraints")
        master.addConstrs(flow_vars.sum('*', j) - flow_vars.sum(j, '*') == 0 for j in used_nodes_list)

        # first/last mile flow constraints
        for node_from in tqdm(used_nodes_list, desc="generating flow constraints for MoD"):
            for node_to in used_nodes_list:
                if node_from != node_to:
                    first_mile_vars_for_from_to = first_mile_vars.get(node_from, {}).get(node_to, [])
                    last_mile_vars_for_from_to = last_mile_vars.get(node_from, {}).get(node_to, [])
                    no_mt_vars_for_from_to = no_mt_vars.get(node_from, {}).get(node_to, [])
                    
                    first_mile_expr = LinExpr([1 for _ in first_mile_vars_for_from_to], first_mile_vars_for_from_to)
                    last_mile_expr = LinExpr([1 for _ in last_mile_vars_for_from_to], last_mile_vars_for_from_to)
                    no_mt_expr = LinExpr([1 for _ in no_mt_vars_for_from_to], no_mt_vars_for_from_to)
                    master.addConstr(
                        flow_vars[node_from, node_to] - first_mile_expr - last_mile_expr - no_mt_expr >= 0,
                        name="first_last_mile[%d][%d]" % (node_from, node_to)
                    )

        output_dir_path = Path(output_dir)

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        if gurobi_log_file is not None:
            gurobi_log_path = Path(gurobi_log_file)
            gurobi_log_path.parent.mkdir(parents=True, exist_ok=True)
            master.Params.LogFile = str(gurobi_log_path)

        if export_model:
            master.write(str(output_dir_path / "ILP.lp"))

        t0 = time.time()
        master.optimize()
        t1 = time.time()

        logging.info("Execution time: %s", t1-t0)
        logging.info("Final solution: %s", master.ObjVal)

        if export_solution:
            master.write(str(output_dir_path / "ILP.sol"))

        self._export_used_lines(
            output_dir=output_dir_path,
            line_vars=line_vars,
            line_costs=line_costs,
        )

        partial_assignments = self._assignments_from_passenger_vars(
            passenger_vars=passenger_vars,
        )
        _request_assignments, assignments = self._resolve_and_export_request_assignments(
            output_dir=output_dir_path,
            request_assignments=partial_assignments,
            no_assignment_handling=NoAssignmentHandling.REJECT,
        )

        flows = {
            (i, j): var.X
            for (i, j), var in flow_vars.items()
            if var.X > EPS and i != j
        }
        self._export_flows(flows, output_dir_path)

        return master.ObjVal, t1-t0

    def rounding(self, solution, active_sets, sets):
        nb_pass = self.line_instance.nb_pass
        nb_lines = self.line_instance.nb_lines
        capacity = self.line_instance.capacity

        lines_cost = [self.cost_coefficient * self.line_instance.lengths_travel_times[l//self.max_frequency] + l % self.max_frequency * self.line_instance.lengths_travel_times[l//self.max_frequency] for l in range(nb_lines * self.max_frequency)]

        values = [0 for p in range(self.line_instance.nb_pass)] #store the value we will get of passengers after the aggregation step

        passenger_assignment  = [] #Useless, for sanity check
        used_budget = 0
        opened_lines = []
        for l in range(len(active_sets)):
            prob = 0
            r = np.random.random()

            final_set_index = None #Initially, no set is assigned to the line
            for s in active_sets[l]: #the candidate sets are such that X[l,s] > 0
                if r <= solution[l,s] + prob:
                    final_set_index = s #final index of the set assigned to l
                    break
                else:
                    prob += solution[l,s]
            if final_set_index: #if final_set_index is false, the line is not opened
                passenger_assignment.append([])
                for p in sets[final_set_index]: #for the passengers in the set of index final_set_index
                    rho = l // self.max_frequency
                    trip_val = self.line_instance.trip_value_on_line(p, rho)
                    if trip_val > values[p]: #if I could get more value by reassigning passenger p to line l
                        values[p] = trip_val
                        passenger_assignment[len(passenger_assignment)-1].append([p,values[p]])

                used_budget += lines_cost[l] #Add costs if line is opened
                opened_lines.append(l)

        total_value = sum(values)

        return used_budget, total_value, opened_lines, passenger_assignment, values

    def execute_proposed_method(
        self,
        Budget,
        candidate_set_of_lines,
        gurobi_subproblem_method: Optional[int] = None,
    ):
        """
        Execute the proposed method with LP solving and rounding iterations to find the best solution.

        Args:
            Budget: Budget constraint
            candidate_set_of_lines: Candidate set of lines
            gurobi_subproblem_method: optional Gurobi ``Method`` for column-generation sub-MIPs.

        Returns:
            tuple: (best_value, used_budget, opened_lines_info, values, nb_respect, mean_value, execution_time)
        """
        logging.info("Solving the line planning problem with the proposed method")
        print("LP")
        solution, active_sets, sets, execution_time = self.solve_master_LP(gurobi_subproblem_method)

        best_value = 0
        budg = 0
        iter = 0
        op = []
        mean = 0
        nb_respect = 0
        v = None
        pass_ass = None  # for sanity check

        np.random.seed(127)
        # Do 10000 iterations of the rounding process and keep the best one
        while iter <= 10000:
            used_budget, value, opened_lines, passenger_assignment, values = (
                self.rounding(solution, active_sets, sets)
            )
            if used_budget <= Budget:
                if value > best_value:
                    best_value = value
                    budg = used_budget
                    v = values
                    pass_ass = passenger_assignment
                    op = [
                        [
                            opened_lines[l] // self.max_frequency,
                            opened_lines[l] % self.max_frequency,
                        ]
                        for l in range(len(opened_lines))
                    ]  # contains [l,f_l] for the lines l opened with frequency f_l
                mean += value
                nb_respect += 1
            iter += 1

        print("best value", best_value, "budget", budg)
        print("nb_respect", nb_respect)
        if nb_respect > 0:
            print("opened", op)
            for l in range(len(op)):
                print("line", l, "nodes", candidate_set_of_lines[op[l][0]])
            tot_assigned = 0
            for p in range(len(values)):
                if values[p] > 0:
                    tot_assigned += 1
            print("nb_assigned", tot_assigned)

        return best_value, budg, op, v, nb_respect, mean, execution_time


def _parse_yaml_bool(raw: object, default: bool = False) -> bool:
    """Coerce experiment YAML values to bool (handles native bool and common string forms)."""
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return raw != 0
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    return default


def _parse_optional_budget(raw: object) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in ("", "none", "null", "inf", "infinity"):
        return None
    return float(raw)


def _ilp_budget_rhs(budget: Optional[float]) -> float:
    if budget is None:
        return GRB.INFINITY
    return float(budget)


def _parse_max_travel_time_delay_seconds(raw: Dict[str, Any], experiment_config_path: Path) -> int:
    mtd = raw.get("max_travel_time_delay")
    if isinstance(mtd, dict) and mtd:
        mode_raw = mtd.get("mode", "absolute")
        mode = str(mode_raw).strip().lower()
        if mode != "absolute":
            raise ValueError(
                f"{experiment_config_path}: max_travel_time_delay.mode must be 'absolute', got {mode_raw!r}."
            )
        sec_raw = mtd.get("seconds")
        if sec_raw is None:
            raise ValueError(
                f"{experiment_config_path}: max_travel_time_delay.seconds is required when "
                "max_travel_time_delay is set."
            )
        return int(sec_raw)
    return 300


def run_experiment(experiment_config_path: Path) -> None:
    """
    Run one line-planning experiment from a YAML file.

    The YAML must define ``instance`` (path to instance ``config.yaml``).
    Preprocessing caches are written under ``<instance_config_directory>/preprocessing/``.
    Optional ``results_dir`` is the output directory for logs, exports, and metrics;
    if omitted, outputs go next to the experiment YAML. Optional ``mass_transport`` /
    ``solver`` blocks and optional ``budget`` (omit for an unconstrained ILP budget).

    ``mass_transport.cost_coefficient`` scales line operating cost in the ILP (default ``1``).
    ``mass_transport.max_frequency`` is the per-route replication / frequency cap (default ``1``).
    Optional ``mass_transport.pruning`` is an ordered list of trip-option pruning method objects.
    Supported methods are ``mt_time_share`` and ``line_mod_aggregate``.

    The ``solver`` block must set ``method`` to exactly one of:
    ``approximation``, ``ilp``, ``ilp_with_mod_costs``, ``ilp_with_empty_trips``,
    ``non_budget_ilp``, ``peak_batch_max_headway_ilp``.
    Optional ``solver.time_limit`` (seconds) is passed to ``LinePlanningSolver`` (Gurobi time limit;
    default ``86400``).

    For ``method: approximation``, optional ``solver.approximation_subproblem_method`` (int)
    sets Gurobi ``Method`` on each column-generation sub-MIP (omit or ``0`` for default).

    For ``method: non_budget_ilp``, optional ``solver.rejection_cost`` (default ``0``) enables
    request rejection in the ILP (§4.1.2); see experiment README.
    Optional ``solver.use_request_line_valid_inequalities`` (default ``false``) adds inequalities
    ``x_{ρr} ≤ y_ρ`` to that ILP only.
    """
    experiment_config_path = experiment_config_path.resolve()
    exp = load_experiment_yaml(experiment_config_path)
    inst_path = resolve_instance_config_path(experiment_config_path, exp)
    inst = load_line_planning_instance_config(inst_path)
    base_results_directory = resolve_results_dir(experiment_config_path, exp)

    solver_cfg = exp.get("solver") or {}
    time_limit = float(solver_cfg.get("time_limit", 3600 * 24))
    rejection_cost_cfg = float(solver_cfg.get("rejection_cost", 0) or 0)
    use_request_line_valid_inequalities = _parse_yaml_bool(
        solver_cfg.get("use_request_line_valid_inequalities"), default=False
    )
    max_travel_time_delay_seconds = _parse_max_travel_time_delay_seconds(exp, experiment_config_path)

    solver_method = _resolve_solver_method(solver_cfg)

    mt = dict(exp.get("mass_transport") or {})
    cost_coefficient = float(mt.get("cost_coefficient", 1))
    max_frequency = int(mt.get("max_frequency", 1))
    try:
        trip_option_pruning = lineplanning.instance.normalize_trip_option_pruning_specs(mt.get("pruning"))
    except ValueError as exc:
        raise ValueError(f"{experiment_config_path}: invalid mass_transport.pruning: {exc}") from exc

    preprocessing_dir = inst.config_path.parent / "preprocessing"
    line_inst = line_instance(
        candidate_lines_file=inst.lines_file,
        capacity=int(mt.get("capacity", 30)),
        maximum_detour=mt.get("maximum_detour", 3),
        demand_file=inst.demand_file,
        preprocessing_dir=preprocessing_dir,
        dm_file=inst.dm_file,
        trip_option_pruning=trip_option_pruning,
    )

    budget = _parse_optional_budget(exp.get("budget"))

    instance_size_label = get_instance_size_label(str(inst.demand_file))
    base_results_directory.mkdir(parents=True, exist_ok=True)

    if budget is not None:
        line_inst.B = budget * 0.95
    else:
        line_inst.B = GRB.INFINITY

    candidate_set_of_lines = line_inst.candidate_set_of_lines
    logging.info("Loaded %d candidate routes.", len(candidate_set_of_lines))
    solver = LinePlanningSolver(
        line_inst,
        time_limit=time_limit,
        cost_coefficient=cost_coefficient,
        max_frequency=max_frequency,
    )

    run_log_path = base_results_directory / "run.log"
    gurobi_log_path = base_results_directory / "gurobi.log"
    log_handler = _configure_run_logging(run_log_path)
    obj_val: Union[float, int] = 0.0
    run_time_sec = 0.0
    line_obj_component: Optional[float] = None
    mod_obj_component: Optional[float] = None
    try:
        if solver_method == "approximation":
            prop_budget = float("inf") if budget is None else float(budget)
            raw_sub_mip = solver_cfg.get("approximation_subproblem_method")
            gurobi_subproblem_method = int(raw_sub_mip) if raw_sub_mip is not None else None
            logging.info("Running solver.method=approximation (budget cap for rounding=%s)", prop_budget)
            best_value, budg, op, v, nb_respect, mean, execution_time = solver.execute_proposed_method(
                prop_budget,
                candidate_set_of_lines,
                gurobi_subproblem_method=gurobi_subproblem_method,
            )
            obj_val = float(best_value)
            run_time_sec = float(execution_time)
            logging.info(
                "Approximation finished: best_value=%s budg=%s nb_respect=%s time=%s",
                best_value,
                budg,
                nb_respect,
                execution_time,
            )
        elif solver_method in ("ilp", "ilp_with_mod_costs", "ilp_with_empty_trips"):
            logging.info("Running solver.method=%s (budget=%s)", solver_method, budget)
            solver.line_instance.B = _ilp_budget_rhs(budget)
            if solver_method == "ilp_with_mod_costs":
                lsr = solver.solve_modified_ILP(
                    export_model=True,
                    export_solution=True,
                    output_dir=base_results_directory,
                    gurobi_log_file=gurobi_log_path,
                )
                obj_val = lsr.objective_value
                run_time_sec = lsr.run_time_seconds
                line_obj_component = lsr.line_objective_component
                mod_obj_component = lsr.mod_objective_component
            elif solver_method == "ilp_with_empty_trips":
                obj_val, run_time_sec = solver.solve_ILP_with_empty_trips(
                    export_model=True,
                    export_solution=True,
                    output_dir=base_results_directory,
                    gurobi_log_file=gurobi_log_path,
                )
            else:
                obj_val, run_time_sec = solver.solve_ILP(
                    export_model=True,
                    export_solution=True,
                    output_dir=base_results_directory,
                    gurobi_log_file=gurobi_log_path,
                )
        elif solver_method == "non_budget_ilp":
            logging.info(
                "Running solver.method=non_budget_ilp (MoD-aware route-aggregated ILP, "
                "rejection_cost=%s, use_request_line_valid_inequalities=%s)",
                rejection_cost_cfg,
                use_request_line_valid_inequalities,
            )
            lsr = solver.solve_MoD_aware_ILP(
                export_model=True,
                export_solution=True,
                output_dir=base_results_directory,
                gurobi_log_file=gurobi_log_path,
                rejection_cost=rejection_cost_cfg,
                use_request_line_valid_inequalities=use_request_line_valid_inequalities,
            )
            obj_val = lsr.objective_value
            run_time_sec = lsr.run_time_seconds
            line_obj_component = lsr.line_objective_component
            mod_obj_component = lsr.mod_objective_component
        elif solver_method == "peak_batch_max_headway_ilp":
            logging.info(
                "Running solver.method=peak_batch_max_headway_ilp (max_wait_time=%s, max_route_frequency=%s)",
                max_travel_time_delay_seconds,
                max_frequency,
            )
            lsr = solver.solve_peak_batch_max_headway_ILP(
                export_model=True,
                export_solution=True,
                output_dir=base_results_directory,
                gurobi_log_file=gurobi_log_path,
                max_route_frequency=max_frequency,
                max_wait_time=max_travel_time_delay_seconds,
            )
            obj_val = lsr.objective_value
            run_time_sec = lsr.run_time_seconds
            line_obj_component = lsr.line_objective_component
            mod_obj_component = lsr.mod_objective_component
        else:
            raise AssertionError("unreachable solver_method")
    finally:
        root_logger = logging.getLogger()
        root_logger.removeHandler(log_handler)
        log_handler.close()

    results_payload: Dict[str, Union[str, float, int, None]] = {
        "objective_value": obj_val,
        "run_time_seconds": run_time_sec,
        "method": solver_method,
        "instance_size": instance_size_label,
        "demand_file": str(inst.demand_file),
        "instance_config": str(inst.config_path),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if line_obj_component is not None:
        results_payload["line_objective_component"] = line_obj_component
    if mod_obj_component is not None:
        results_payload["mod_objective_component"] = mod_obj_component
    if solver_method == "non_budget_ilp":
        if rejection_cost_cfg > 0:
            results_payload["rejection_cost"] = rejection_cost_cfg
        if use_request_line_valid_inequalities:
            results_payload["use_request_line_valid_inequalities"] = True
    if solver_method == "peak_batch_max_headway_ilp":
        results_payload["max_wait_time"] = max_travel_time_delay_seconds
    if budget is not None:
        results_payload["budget"] = budget
    results_file = base_results_directory / "metrics.json"
    results_file.write_text(json.dumps(results_payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Line planning experiment driver (YAML experiment references instance config.yaml).",
    )
    parser.add_argument(
        "experiment_config",
        type=Path,
        help="Path to experiment YAML (must set 'instance'; optional 'results_dir').",
    )
    args = parser.parse_args()
    run_experiment(args.experiment_config)
