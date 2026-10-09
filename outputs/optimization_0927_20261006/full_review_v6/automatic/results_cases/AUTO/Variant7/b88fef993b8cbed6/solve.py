import gurobipy as gp
from gurobipy import GRB
sources = ['S1', 'S2', 'S3']
hubs = ['H1', 'H2']
customers = ['C1', 'C2', 'C3', 'C4']
source_supply = {'S1': 120, 'S2': 100, 'S3': 90}
customer_demand = {'C1': 70, 'C2': 80, 'C3': 60, 'C4': 90}
hub_capacity = {'H1': 170, 'H2': 160}
arc_cost = {('S1', 'H1'): 2, ('S1', 'H2'): 6, ('S2', 'H1'): 4, ('S2', 'H2'): 3, ('S3', 'H1'): 7, ('S3', 'H2'): 2, ('H1', 'C1'): 3, ('H1', 'C2'): 4, ('H1', 'C3'): 7, ('H1', 'C4'): 8, ('H2', 'C1'): 8, ('H2', 'C2'): 6, ('H2', 'C3'): 3, ('H2', 'C4'): 4}
for s in sources:
    for h in hubs:
        if (s, h) not in arc_cost:
            raise ValueError(f'Missing arc cost for ({s},{h})')
for h in hubs:
    for c in customers:
        if (h, c) not in arc_cost:
            raise ValueError(f'Missing arc cost for ({h},{c})')
for s in sources:
    if s not in source_supply:
        raise ValueError(f'Missing supply for {s}')
for c in customers:
    if c not in customer_demand:
        raise ValueError(f'Missing demand for {c}')
for h in hubs:
    if h not in hub_capacity:
        raise ValueError(f'Missing capacity for {h}')
source_hub_arcs = [(s, h) for s in sources for h in hubs]
hub_cust_arcs = [(h, c) for h in hubs for c in customers]
all_arcs = source_hub_arcs + hub_cust_arcs
f_vars = gp.Model('DistributionNetwork')
f_vars.Params.MIPGap = 0.0001
f_arc_vars = f_vars.addVars(all_arcs, lb=0, vtype=GRB.CONTINUOUS, name='')
f_vars.setObjective(gp.quicksum((arc_cost[i_j] * f_arc_vars[i_j] for i_j in all_arcs)), GRB.MINIMIZE)
for s in sources:
    f_vars.addConstr(gp.quicksum((f_arc_vars[s, h] for h in hubs)) <= source_supply[s], name=f'supply_{s}')
for c in customers:
    f_vars.addConstr(gp.quicksum((f_arc_vars[h, c] for h in hubs)) >= customer_demand[c], name=f'demand_{c}')
for h in hubs:
    f_vars.addConstr(gp.quicksum((f_arc_vars[s, h] for s in sources)) == gp.quicksum((f_arc_vars[h, c] for c in customers)), name=f'flowbal_{h}')
for h in hubs:
    f_vars.addConstr(gp.quicksum((f_arc_vars[s, h] for s in sources)) <= hub_capacity[h], name=f'hubcap_{h}')
m = f_vars
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')