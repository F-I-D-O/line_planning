def generate_new_line(bus_nodes, cost_edges, min_start_end_distance, detour_skeleton, bus_network, max_travel, max_travel_actual, detour_coeff):
        

        
    remaining_stops = copy.deepcopy(bus_nodes)
    n = len(remaining_stops)
    min_distance = min_start_end_distance


    start_index = random.randint(0,n-1)
    start = remaining_stops.pop(start_index)

    distance_start_end = 0
    i = 0
    while i <= 1000:
        i+=1
        end_index = random.randint(0,n-2)
        end = remaining_stops[end_index]
#         if cost_edges[start, end] >= min_distance:
#             break
            
        if cost_edges[start,end] >= min_distance and cost_edges[start,end] <= max_travel:
            routeA = nx.shortest_path(bus_network, start, end, weight='length')
            if len(set(routeA)) == len(routeA):
                break
            
    end = remaining_stops.pop(end_index)

        
    i = 0
    while i <= 1000:
        i+=1
        inter_index = random.randint(0,n-3)
        inter = remaining_stops[inter_index]
        if cost_edges[start, inter]  + cost_edges[inter, end] <= cost_edges[start, end] * detour_skeleton:
            routeB = nx.shortest_path(bus_network, start, inter, weight='length')
            routeC = nx.shortest_path(bus_network, inter, end, weight='length')
            unionBC = set(routeB) | set(routeC)
            if len(set(unionBC)) == len(set(routeB)) + len(set(routeC)) - 1 and len(set(routeB)) == len(routeB) and len(set(routeC)) == len(routeC):
                break
            
    inter = remaining_stops.pop(inter_index)
    

    i = 0
    while i <= 1000:
        i+=1
        inter_index_2 = random.randint(0,n-4)
        inter_2 = remaining_stops[inter_index_2]
        if cost_edges[inter, inter_2]  + cost_edges[inter_2, end] <= cost_edges[inter, end] * detour_skeleton:
            routeD = nx.shortest_path(bus_network, inter, inter_2, weight='length')
            routeE = nx.shortest_path(bus_network, inter_2, end, weight='length')
            unionDE = set(routeD) | set(routeE)
            if len(set(unionDE)) == len(set(routeD)) + len(set(routeE)) - 1 and len(set(routeD)) == len(routeD) and len(set(routeE)) == len(routeE):
                break
    inter_2 = remaining_stops.pop(inter_index_2)


    # if we use g here we will remove extra nodes not in taxi network later
    route1 = nx.shortest_path(bus_network, start, inter, weight='length')
    route2 = nx.shortest_path(bus_network, inter, inter_2, weight='length')
    route3 = nx.shortest_path(bus_network, inter_2, end, weight='length')
    

    route = route1 + route2[1:] + route3[1:] 
    

    

    # we remove nodes not in taxi network, if needed
    route_temp = []
    for i in range(len(route)):
        if route[i] in bus_nodes:
            route_temp.append(route[i])
    route = copy.deepcopy(route_temp)
    
        
        
#     while len(route) > length_limit:
#         index = random.randint(1,len(route)-2)
#         route.pop(index)


    # output part of the line if two-way line, not longer than the required max length
    route_temp = [route[0]]
    total_length = 0
    for i in range(len(route)-1):
        if total_length + cost_edges[route[i], route[i+1]] + cost_edges[route[i+1], route[i]] <= max_travel_actual:
            route_temp.append(route[i+1])
            total_length += cost_edges[route[i], route[i+1]] + cost_edges[route[i+1], route[i]]
        else:
            break

            

            
    route = route_temp
    
    
    # add detour and subtour constraint for the lines
    detour_violation = False
    subtour_violation = False
    
    route_length = 0
    route_rev_length = 0
    for i in range(len(route)-1):
        route_length += cost_edges[route[i], route[i+1]]
        route_rev_length += cost_edges[route[i+1], route[i]]
    if route_length > detour_coeff*cost_edges[route[0], route[-1]] or route_rev_length > detour_coeff*cost_edges[route[-1], route[0]]:
        detour_violation = True
    if len(set(route)) != len(route):
        subtour_violation = True
        
        
     
    route_rev = route.copy()
    route_rev.reverse() 
    route = route + route_rev[1:]
    
    
    

    
    
    return route, detour_violation, subtour_violation, start, inter, inter_2, end
    
    