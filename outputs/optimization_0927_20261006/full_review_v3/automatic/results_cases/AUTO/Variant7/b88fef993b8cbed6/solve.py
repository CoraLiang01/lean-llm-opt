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
m = gp.Model('DistributionNetwork')
source_hub_arcs = [(s, h) for s in sources for h in hubs]
hub_customer_arcs = [(h, c) for h in hubs for c in customers]
f_source_hub_vars = m.addVars(source_hub_arcs, lb=0, vtype=GRB.CONTINUOUS, name='')
f_hub_customer_vars = m.addVars(hub_customer_arcs, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((arc_cost[s_h] * f_source_hub_vars[s_h] for s_h in source_hub_arcs)) + gp.quicksum((arc_cost[h_c] * f_hub_customer_vars[h_c] for h_c in hub_customer_arcs)), GRB.MINIMIZE)
for s in sources:
    m.addConstr(gp.quicksum((f_source_hub_vars[s, h] for h in hubs)) <= source_supply[s], name=f'supply_{s}')
for c in customers:
    m.addConstr(gp.quicksum((f_hub_customer_vars[h, c] for h in hubs)) >= customer_demand[c], name=f'demand_{c}')
for h in hubs:
    m.addConstr(gp.quicksum((f_source_hub_vars[s, h] for s in sources)) == gp.quicksum((f_hub_customer_vars[h, c] for c in customers)), name=f'flowbal_{h}')
for h in hubs:
    m.addConstr(gp.quicksum((f_source_hub_vars[s, h] for s in sources)) <= hub_capacity[h], name=f'cap_{h}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')