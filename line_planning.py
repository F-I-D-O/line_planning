import logging
import numpy as np
from gurobipy import *
from instance import *
import time
from copy import deepcopy

import log


EPS = 1.e-5
allowed_time = 1200


fixed_cost = 1
max_frequency = 1

class line_planning_solver:

    def __init__(self, line_instance):
        self.line_instance = line_instance

    # Implementation of the column generation process. Outputs the solution of the configuration LP before rounding.
    def solve_master_LP(self):

        nb_pass = self.line_instance.nb_pass
        nb_lines = self.line_instance.nb_lines
        print(nb_pass, nb_lines)

        capacity = self.line_instance.capacity #capacity of a line

        t_1 = time.time()

        sets = [] #set[j] stores the indices of passengers present in the set of index j
        lines_to_sets = []	# contains, for each line l', the indices of the active sets of passengers
        for j in range(nb_lines * max_frequency):
            lines_to_sets.append([])

        # cost proportional to travel time on line
        lines_cost = [fixed_cost * self.line_instance.lengths_travel_times[l//max_frequency] + l % max_frequency * self.line_instance.lengths_travel_times[l//max_frequency] for l in range(nb_lines * max_frequency)]

        passengers_to_sets = [[[] for l in range(nb_lines * max_frequency)] for p in range(nb_pass)] #for passenger index p and line l, contains the list of indices of sets including passenger p

        # generate initial sets with one passenger covered per line (if no passenger covered, add empty set)
        for l in range(len(lines_to_sets)):
            route = l // max_frequency
            if len(self.line_instance.set_of_lines[route][1]) != 0:
                p = self.line_instance.set_of_lines[route][1][0][0]
                sets.append([p])
                lines_to_sets[l].append(l)
                passengers_to_sets[p][l].append(l)
            else:
                sets.append([])
                lines_to_sets[l].append(l)

        lines_to_passengers = []
        for l in range(len(lines_to_sets)):
            lines_to_passengers.append(self.line_instance.lines_to_passengers[l//max_frequency])

        covered_average = 0
        iter = 0

        # Build master LP
        master = Model("LP")
        master.ModelSense = -1

        # Define variables
        x = {}
        for l in range(len(lines_to_sets)):
            for s in lines_to_sets[l]:
                total_set_value = sum([self.line_instance.optimal_trip_options[p][l // max_frequency].value for p in sets[s]])
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

        master.Params.timeLimit = allowed_time

        master.optimize()

        t_4 = 0

        x_temp = {}
        for l in range(len(lines_to_sets)):
            for s in lines_to_sets[l]:
                x_temp[l,s] = deepcopy(x[l,s].X)

        obj_temp = master.ObjVal
        lines_to_sets_temp = deepcopy(lines_to_sets)

        while t_4 - t_2 <= allowed_time:
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
                    f_l = l%max_frequency + 1
                    length = self.line_instance.set_of_lines[l//max_frequency][0]

                    single_line = Model("SLP") #single line sub-problem
                    single_line.ModelSense = -1 #maximize

                    if self.line_instance.method != 0:
                        single_line.Params.OutputFlag = 0
                        single_line.Params.Method = self.line_instance.method

                    y = {}
                    for p in lines_to_passengers[l]:
                        y[p] = single_line.addVar(obj=self.line_instance.optimal_trip_options[p][l // max_frequency].value - lamb[p], ub=1, vtype=GRB.BINARY, name="y[%d]" % p)
                    single_line.update()
                    var = [y[p] for p in lines_to_passengers[l]]

                    for k in range(length):
                        coef = []
                        edges_to_pass = self.line_instance.edge_to_passengers[l//max_frequency][k]
                        index = 0
                        for p in lines_to_passengers[l]:
                            if index < len(edges_to_pass) and edges_to_pass[index] == p:
                                coef.append(1)
                                index += 1
                            else:
                                coef.append(0)
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

                        total_set_value = sum([self.line_instance.optimal_trip_options[p][l // max_frequency].value for p in sets[s]])
                        x[l,s] = master.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj = total_set_value, name="x[%d][%d]"%(l,s), column = col)
                        t_update = time.time()

            if not found_a_new_set: #no more columns need to be added
                break
            master.update()
            t_3= time.time()

            if t_3 - t_2 <= allowed_time:
                master.Params.timeLimit = allowed_time - t_3 + t_2
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

    def solve_ILP(self):
        request_count = self.line_instance.nb_pass
        line_count = self.line_instance.nb_lines
        bus_capacity = self.line_instance.capacity

        line_count_total = line_count * max_frequency

        lines_cost = [
            fixed_cost * self.line_instance.lengths_travel_times[l // max_frequency]
            + l
            % max_frequency
            * self.line_instance.lengths_travel_times[l // max_frequency]
            for l in range(line_count * max_frequency)
        ]

        master = Model("LP") # master LP problem
        master.ModelSense = -1 # maximize the objective function

        master.Params.timeLimit = allowed_time

        # Line variables
        y = {} # binary variable indicating if line l is opened
        for l in range(line_count_total):
            y[l] = master.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.BINARY, obj = 0, name="y[%d]"%l)
        master.update()

        # Passenger variables
        x = {}
        for l in range(line_count_total):
            for p in range(request_count):
                val = self.line_instance.optimal_trip_options[p][l // max_frequency].value
                if val > 0:
                    x[l,p] = master.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.BINARY, obj = val, name="x[%d][%d]"%(l,p))
        master.update()

        # One line per passenger constraints
        one_line_per_passenger = {}
        for p in range(request_count):
            var = []
            for l in range(line_count_total):
                val = self.line_instance.optimal_trip_options[p][l // max_frequency].value
                if val > 0:
                    var.append(x[l,p])
            coef = [1 for j in range(len(var))]
            one_line_per_passenger[p] = master.addConstr(LinExpr(coef,var) <= 1, name="one_line_per_passenger[%d]"%p)
        master.update()

        # Bus capacity constraints
        capacity_constraints = {}
        for l in range(line_count_total):
            f_l = l%max_frequency + 1
            length = self.line_instance.set_of_lines[l//max_frequency][0]
            for k in range(length):
                var = []
                coef = []
                for p in self.line_instance.edge_to_passengers[l//max_frequency][k]:
                    var.append(x[l,p])
                    coef.append(1)
                capacity_constraints[l,k] = master.addConstr(LinExpr(coef,var) <= bus_capacity * f_l * y[l], name="capacity_constraints[%d][%d]"%(l,k))
        master.update()

        # Budget constraint
        var = [y[l] for l in range(line_count_total)]
        coef = [lines_cost[l] for l in range(line_count_total)]
        budget_constraint = master.addConstr(LinExpr(coef,var) <= self.line_instance.B, name="budget_constraints")
        master.update()

        # Model parameters
        master.Params.Presolve = 0
        master.Params.PreCrush = 1
        master.Params.Cuts = 0

        logging.info('method: %s', master.Params.Method)
        t0 = time.time()
        master.optimize()
        t1 = time.time()
        logging.info("Execution time: %s", t1-t0)
        logging.info("Final solution: %s", master.ObjVal)
        for l in range(line_count_total):
            if y[l].X>0:
                op = [l//max_frequency,l%max_frequency]
                print("line", l, y[l].X, 'cost', lines_cost[l])
                print(op)
                '''for p in range(nb_pass):
                    if(self.line_instance.values[p][l // max_frequency]>0):
                        print("passenger", p, x[l,p].X, self.line_instance.values[p][l // max_frequency])'''

        return master.ObjVal, t1-t0

    def solve_modified_ILP(self, export_model=False, export_solution=False):
        request_count = self.line_instance.nb_pass
        line_count = self.line_instance.nb_lines
        bus_capacity = self.line_instance.capacity

        line_count_total = line_count * max_frequency

        master = Model("Modified ILP") # master LP problem
        master.ModelSense = -1 # maximize the objective function

        master.Params.timeLimit = allowed_time

        # binary variables indicating if line l is opened
        line_vars = master.addVars(line_count_total, vtype=GRB.BINARY, name="y")

        # binary variables indicating if passenger p is assigned to line l. If first mile + last mile costs are
        # higher than the no_MT MoD cost, the line-passenger combination is not considered at all
        potential_line_passenger_combinations = [
            (l, p) for l in range(line_count_total) for p in range(request_count)
            if self.line_instance.optimal_trip_options[p][l // max_frequency].value > 0
        ]
        passenger_vars = master.addVars(potential_line_passenger_combinations, vtype=GRB.BINARY, obj=1, name="x")
        # add no MT variables for each passenger
        for p in range(request_count):
            passenger_vars[line_count_total, p] = master.addVar(vtype=GRB.BINARY, obj=1, name="x[no_MT,%d]" % p)

        # One line per passenger constraints
        master.addConstrs(
            (passenger_vars.sum('*', p) <= 1 for p in range(request_count)),
            name="one_line_per_passenger"
        )

        # Bus capacity constraints
        capacity_constraints = {}
        for l in range(line_count_total):
            f_l = l%max_frequency + 1
            length = self.line_instance.set_of_lines[l//max_frequency][0]
            for k in range(length):
                vars = []
                coefs = []
                for p in self.line_instance.edge_to_passengers[l//max_frequency][k]:
                    vars.append(passenger_vars[l,p])
                    coefs.append(1)
                capacity_constraints[l,k] = master.addConstr(LinExpr(coefs,vars) <= bus_capacity * f_l * line_vars[l], name="capacity_constraints[%d][%d]"%(l,k))

        # Budget constraint
        line_costs = [
            fixed_cost * self.line_instance.lengths_travel_times[l // max_frequency]
            + l
            % max_frequency
            * self.line_instance.lengths_travel_times[l // max_frequency]
            for l in range(line_count_total)
        ]
        line_costs_expression = line_vars.prod(line_costs)
        mod_costs = {}
        for p in range(request_count):
            for l in range(line_count_total + 1):
                optimal_trip_option: TripOption = self.line_instance.optimal_trip_options[p][l // max_frequency]
                mod_costs[l,p] = optimal_trip_option.first_mile_cost + optimal_trip_option.last_mile_cost
        mod_cost_expression = passenger_vars.prod(mod_costs)
        master.addConstr(line_costs_expression + mod_cost_expression <= self.line_instance.B, name="budget_constraint")

        # Model parameters
        master.Params.Presolve = 0
        master.Params.PreCrush = 1
        master.Params.Cuts = 0

        if export_model:
            master.write("modified_ILP.lp")

        t0 = time.time()
        master.optimize()
        t1 = time.time()

        logging.info("Execution time: %s", t1-t0)
        logging.info("Final solution: %s", master.ObjVal)

        if export_solution:
            master.write("modified_ILP.sol")

        for l in range(line_count_total):
            if line_vars[l].X>0:
                op = [l//max_frequency,l%max_frequency]
                print("line", l, line_vars[l].X, 'cost', line_costs[l])
                print(op)
                '''for p in range(nb_pass):
                    if(self.line_instance.values[p][l // max_frequency]>0):
                        print("passenger", p, x[l,p].X, self.line_instance.values[p][l // max_frequency])'''

        return master.ObjVal, t1-t0

    def solve_ILP_with_empty_trips(self, export_model=False, export_solution=False):
        request_count = self.line_instance.nb_pass
        line_count = self.line_instance.nb_lines
        bus_capacity = self.line_instance.capacity
        node_count = len(self.line_instance.dm)

        line_count_total = line_count * max_frequency

        logging.info("Building ILP model with empty trips")

        master = Model("ILP with empty trips") # master LP problem
        master.ModelSense = -1 # maximize the objective function

        master.Params.timeLimit = allowed_time

        # VARIABLES
        # binary variables indicating if line l is opened
        line_vars = master.addVars(line_count_total, vtype=GRB.BINARY, name="y")

        # binary variables indicating if passenger p is assigned to line l. If first mile + last mile costs are
        # higher than the no_MT MoD cost, the line-passenger combination is not considered at all
        potential_line_passenger_combinations = [
            (l, p) for l in range(line_count_total) for p in range(request_count)
            if self.line_instance.optimal_trip_options[p][l // max_frequency].value > 0
        ]
        passenger_vars = master.addVars(potential_line_passenger_combinations, vtype=GRB.BINARY, obj=1, name="x")
        # add no MT variables for each passenger
        for p in range(request_count):
            passenger_vars[line_count_total, p] = master.addVar(vtype=GRB.BINARY, obj=1, name="x[no_MT,%d]" % p)

        # integer flow variables indicating the MoD flow on each edge of the complete graph
        # first compute used nodes
        used_nodes = set()
        for p in range(request_count):
            used_nodes.add(self.line_instance.requests[p][0])
            used_nodes.add(self.line_instance.requests[p][1])
            for l in range(line_count_total):
                optimal_trip_option: TripOption = self.line_instance.optimal_trip_options[p][l // max_frequency]
                used_nodes.add(optimal_trip_option.mt_pickup_node)
                used_nodes.add(optimal_trip_option.mt_drop_off_node)
        used_nodes_list = list(used_nodes)
        flow_vars = master.addVars(used_nodes_list, used_nodes_list, vtype=GRB.INTEGER, name="phi")

        # One line per passenger constraints
        master.addConstrs(
            (passenger_vars.sum('*', p) <= 1 for p in range(request_count)),
            name="one_line_per_passenger"
        )

        # Bus capacity constraints
        capacity_constraints = {}
        for l in range(line_count_total):
            f_l = l%max_frequency + 1
            length = self.line_instance.set_of_lines[l//max_frequency][0]
            for k in range(length):
                vars = []
                coefs = []
                for p in self.line_instance.edge_to_passengers[l//max_frequency][k]:
                    vars.append(passenger_vars[l,p])
                    coefs.append(1)
                capacity_constraints[l,k] = master.addConstr(LinExpr(coefs,vars) <= bus_capacity * f_l * line_vars[l], name="capacity_constraints[%d][%d]"%(l,k))

        # Budget constraint
        line_costs = [
            fixed_cost * self.line_instance.lengths_travel_times[l // max_frequency]
            + l
            % max_frequency
            * self.line_instance.lengths_travel_times[l // max_frequency]
            for l in range(line_count_total)
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
        master.addConstrs(flow_vars.sum('*', j) - flow_vars.sum(j, '*') == 0 for j in used_nodes_list)

        # first/last mile flow constraints
        for node_from in used_nodes_list:
            for node_to in used_nodes_list:
                if node_from != node_to:
                    first_mile_vars = []
                    last_mile_vars = []
                    no_mt_vars = []
                    for p in range(request_count):
                        if self.line_instance.requests[p][0] == node_from:
                            for l in range(line_count_total):
                                if (l,p) in passenger_vars:
                                    optimal_trip_option: TripOption = self.line_instance.optimal_trip_options[p][l // max_frequency]
                                    if optimal_trip_option.mt_pickup_node == node_to:
                                        first_mile_vars.append(passenger_vars[l,p])
                            if self.line_instance.requests[p][1] == node_to:
                                no_mt_vars.append(passenger_vars[line_count_total,p])
                        if self.line_instance.requests[p][1] == node_to:
                            for l in range(line_count_total):
                                if (l,p) in passenger_vars:
                                    optimal_trip_option: TripOption = self.line_instance.optimal_trip_options[p][l // max_frequency]
                                    if optimal_trip_option.mt_drop_off_node == node_from:
                                        last_mile_vars.append(passenger_vars[l,p])

                    first_mile_expr = LinExpr([1 for _ in first_mile_vars], first_mile_vars)
                    last_mile_expr = LinExpr([1 for _ in last_mile_vars], last_mile_vars)
                    no_mt_expr = LinExpr([1 for _ in no_mt_vars], no_mt_vars)
                    master.addConstr(
                        flow_vars[node_from, node_to] - first_mile_expr - last_mile_expr - no_mt_expr >= 0,
                        name="first_last_mile[%d][%d]" % (node_from, node_to)
                    )

        # Model parameters
        master.Params.Presolve = 0
        master.Params.PreCrush = 1
        master.Params.Cuts = 0

        if export_model:
            master.write("ILP_with_empty_trips.lp")

        t0 = time.time()
        master.optimize()
        t1 = time.time()

        logging.info("Execution time: %s", t1-t0)
        logging.info("Final solution: %s", master.ObjVal)

        if export_solution:
            master.write("ILP_with_empty_trips.sol")

        for l in range(line_count_total):
            if line_vars[l].X>0:
                op = [l//max_frequency,l%max_frequency]
                print("line", l, line_vars[l].X, 'cost', line_costs[l])
                print(op)
                '''for p in range(nb_pass):
                    if(self.line_instance.values[p][l // max_frequency]>0):
                        print("passenger", p, x[l,p].X, self.line_instance.values[p][l // max_frequency])'''

        return master.ObjVal, t1-t0

    def rounding(self, solution, active_sets, sets):
        optimal_trip_options: List[List[TripOption]] = self.line_instance.optimal_trip_options
        nb_pass = self.line_instance.nb_pass
        nb_lines = self.line_instance.nb_lines
        capacity = self.line_instance.capacity

        lines_cost = [fixed_cost * self.line_instance.lengths_travel_times[l//max_frequency] + l % max_frequency * self.line_instance.lengths_travel_times[l//max_frequency] for l in range(nb_lines * max_frequency)]

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
                    if optimal_trip_options[p][l//max_frequency].value > values[p]: #if I could get more value by reassigning passenger p to line l
                        values[p] = optimal_trip_options[p][l//max_frequency].value
                        passenger_assignment[len(passenger_assignment)-1].append([p,values[p]])

                used_budget += lines_cost[l] #Add costs if line is opened
                opened_lines.append(l)

        total_value = sum(values)

        return used_budget, total_value, opened_lines, passenger_assignment, values

    def execute_proposed_method(self, Budget, candidate_set_of_lines):
        """
        Execute the proposed method with LP solving and rounding iterations to find the best solution.
        
        Args:
            Budget: Budget constraint
            candidate_set_of_lines: Candidate set of lines
            
        Returns:
            tuple: (best_value, used_budget, opened_lines_info, values, nb_respect, mean_value, execution_time)
        """
        logging.info("Solving the line planning problem with the proposed method")
        print("LP")
        solution, active_sets, sets, execution_time = self.solve_master_LP()

        best_value = 0
        budg = 0
        iter = 0
        op = []
        mean = 0
        nb_respect = 0
        v = None
        pass_ass = None  # for sanity check

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
                            opened_lines[l] // max_frequency,
                            opened_lines[l] % max_frequency,
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

if __name__ == "__main__":

    month = "april"
    # month = 'march'
    # month = 'feb'
    np.random.seed(127)
    demand_file = "OD_matrix_april_fhv_1_percent.txt" # override the month setting and uses a custom demand file
    # demand_file = "OD_matrix_april_fhv_10_percent.txt" # override the month setting and uses a custom demand file
    # demand_file = None
    run_proposed_method = False
    use_model_with_mod_costs = False # stage 1 model
    use_model_with_empty_trips = True # stage 2 model

    average_value_LP = 0
    average_value_ILP = 0
    obj_val = 0
    time_LP = 0
    time_ILP = 0

    max_frequency = 1
    nb_l = 10
    # nb_p = 13847
    # nb_p = 12301
    # nb_p = 13851
    nb_p = 130
    # nb_p = 1300

    logging.info("Solving Manhattan instance with %s lines and %s passengers", nb_l, nb_p)

    Budget = 0
    line_inst = line_instance(
        nb_lines=nb_l,
        nb_pass=nb_p,
        B=0.95 * Budget,
        cost=1,
        max_length=15,
        min_length=8,
        proba=0.1,
        capacity=30,
        instance_category="manhattan",
        detour_factor=3,
        method=3,
        granularity=1,
        date=month,
        demand_file=demand_file,
    )

    for ind in range(0, 1):
        if ind == 0:
            # Budget = 100
            Budget = 10000
            max_frequency = 1
            line_inst.B = Budget * 0.95
        if ind == 1:
            Budget = 20000
            max_frequency = 1
            line_inst.B = Budget * 0.95
        if ind == 2:
            Budget = 30000
            max_frequency = 1
            line_inst.B = Budget * 0.95
        if ind == 3:
            Budget = 40000
            max_frequency = 1
            line_inst.B = Budget * 0.95
        if ind == 4:
            Budget = 50000
            max_frequency = 1
            line_inst.B = Budget * 0.95
        if ind == 5:
            Budget = 100000
            max_frequency = 1
            line_inst.B = Budget * 0.95
        if ind == 6:
            Budget = 200000
            max_frequency = 1
            line_inst.B = Budget * 0.95
        print("iter", ind, Budget, max_frequency)

        candidate_set_of_lines = line_inst.candidate_set_of_lines
        print("candidate_set_of_lines")
        solver = line_planning_solver(line_inst)

        # Execute the proposed method with LP solving and rounding
        if run_proposed_method:
            best_value, budg, op, v, nb_respect, mean, execution_time = solver.execute_proposed_method(
                Budget, candidate_set_of_lines
            )
            time_LP += execution_time
            average_value_LP += best_value

        # Solve the line planning problem with ILP
        # try:
        logging.info("Solving the line planning problem with ILP")
        solver.line_instance.B = Budget
        if use_model_with_mod_costs:
            obj_val, run_time_ILP = solver.solve_modified_ILP(export_model=True, export_solution=True)
        elif use_model_with_empty_trips:
            obj_val, run_time_ILP = solver.solve_ILP_with_empty_trips(export_model=True, export_solution=True)
        else:
            obj_val, run_time_ILP = solver.solve_ILP()
        # except Exception as e:
        #     logging.error("error with ILP: %s", e)
