"""
Lightweight checks for peak-batch max-headway helper behavior (no Gurobi solve).
"""

import csv
import importlib.util
import tempfile
from pathlib import Path

import pandas as pd

from lineplanning.line_planning import LinePlanningSolver, NoAssignmentHandling


class FakeVar:
    def __init__(self, value):
        self.X = value


class FakeLineInstance:
    def __init__(self):
        self.nb_pass = 5
        self.demand = pd.DataFrame({"time": [0, 20, 310, 330, 700]})
        self._direct_costs = [5, 10, 7, 20, 9]
        self._route_costs = {
            (0, 2): 4,
            (1, 2): 8,
            (3, 2): 12,
            (4, 1): 9,
        }

    def direct_trip_mod_cost(self, passenger_idx):
        return self._direct_costs[int(passenger_idx)]

    def has_trip_option_on_line(self, passenger_idx, line_idx):
        return (int(passenger_idx), int(line_idx)) in self._route_costs

    def trip_mod_cost_on_line(self, passenger_idx, line_idx):
        return self._route_costs[(int(passenger_idx), int(line_idx))]


def _solver_with_fake_instance():
    solver = LinePlanningSolver.__new__(LinePlanningSolver)
    solver.line_instance = FakeLineInstance()
    solver.max_frequency = 3
    solver.line_count_total = 99
    return solver


def test_peak_batch_tie_breaks_to_earliest_batch():
    solver = _solver_with_fake_instance()
    assert solver._peak_batch_request_indices(300) == ([0, 1], 0)


def test_non_peak_assignment_uses_cheapest_selected_route_or_direct_mod():
    solver = _solver_with_fake_instance()
    assignments = solver._assign_remaining_requests_to_selected_routes(
        selected_routes=[2, 1],
        fixed_assignments={0: ("no_MT", None)},
    )
    assert assignments == [
        ("no_MT", None),
        ("line", 2),
        ("no_MT", None),
        ("line", 2),
        ("no_MT", None),
    ]


def test_variable_assignment_extraction_supports_route_agg_and_rejection():
    solver = _solver_with_fake_instance()
    assignments = solver._assignments_from_passenger_vars(
        passenger_vars={
            (2, 0): FakeVar(1),
            (10, 1): FakeVar(1),
            (3, 2): FakeVar(1),
        },
        no_mt_line_key=10,
        line_var_is_route_index=True,
        rejection_vars={2: FakeVar(1)},
    )
    assert assignments == {
        0: ("line", 2),
        1: ("no_MT", None),
        2: ("rejected", None),
    }


def test_variable_assignment_extraction_supports_encoded_line_indices():
    solver = _solver_with_fake_instance()
    assignments = solver._assignments_from_passenger_vars(
        passenger_vars={(7, 0): FakeVar(1)},
    )
    assert assignments == {0: ("line", 2)}


def test_shared_assignment_resolution_modes_and_export():
    solver = _solver_with_fake_instance()
    with tempfile.TemporaryDirectory() as tmp:
        full, df = solver._resolve_and_export_request_assignments(
            output_dir=Path(tmp),
            request_assignments={0: ("line", 2)},
            no_assignment_handling=NoAssignmentHandling.NO_MT,
        )
        assert full[0] == ("line", 2)
        assert full[1:] == [("no_MT", None)] * 4
        assert df.loc[0, "line_repr"] == 2
        assert df.loc[1, "line_repr"] == "no_MT"
        rows = list(csv.DictReader((Path(tmp) / "passenger_assignments.csv").open()))
        assert rows[0]["line"] == "2"
        assert rows[1]["line"] == "no_MT"

    with tempfile.TemporaryDirectory() as tmp:
        full, _df = solver._resolve_and_export_request_assignments(
            output_dir=Path(tmp),
            request_assignments={},
            no_assignment_handling=NoAssignmentHandling.REJECT,
        )
        assert full == [("rejected", None)] * 5

    with tempfile.TemporaryDirectory() as tmp:
        try:
            solver._resolve_and_export_request_assignments(
                output_dir=Path(tmp),
                request_assignments={},
                no_assignment_handling=NoAssignmentHandling.RAISE,
            )
        except ValueError as exc:
            assert "Missing assignment for passenger 0" in str(exc)
        else:
            raise AssertionError("Expected missing assignment to raise")


def test_loader_rejects_dropped_assignment_value():
    module_path = Path(__file__).resolve().parent / "MoD-aware_line_selection.py"
    spec = importlib.util.spec_from_file_location("mod_aware_line_selection_for_peak_test", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "passenger_assignments.csv"
        path.write_text("passenger,line,mod_cost\n0,Dropped,0\n", encoding="utf-8")
        try:
            mod.load_request_assignments_csv(path)
        except ValueError as exc:
            assert "Invalid assignment value" in str(exc)
        else:
            raise AssertionError("Expected Dropped assignment to raise")


if __name__ == "__main__":
    test_peak_batch_tie_breaks_to_earliest_batch()
    test_non_peak_assignment_uses_cheapest_selected_route_or_direct_mod()
    test_variable_assignment_extraction_supports_route_agg_and_rejection()
    test_variable_assignment_extraction_supports_encoded_line_indices()
    test_shared_assignment_resolution_modes_and_export()
    test_loader_rejects_dropped_assignment_value()
    print("peak-batch max-headway sanity checks passed")
