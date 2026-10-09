import gurobipy as gp
from gurobipy import GRB
sources = ['S1', 'S2', 'S3']
hubs = ['H1', 'H2']
customers = ['C1', 'C2', 'C3', 'C4']
source_supply = {'S1': 120, 'S2': 100, 'S3': 90}
customer_demand = {'C1': 70, 'C2': 80, 'C3': 60, 'C4': 90}
hub_capacity = {'H1': 170, 'H2': 160}
arc_cost = {('S1', 'H1'): 2, ('S1', 'H2'): 6, ('S2', 'H1'): 4, ('S2', 'H2'): 3, ('S3', 'H1'): 7, ('S3', 'H2'): 2, ('H1', 'C1'): 3, ('H1', 'C2'): 4, ('H1', 'C3'): 7, ('H1', 'C4'): 8, ('H2', 'C1'): 8, ('H2', 'C2'): 6, ('H2', 'C3'): 3, ('H2', 'C4'): 4}
expected_arcs = []
for s in sources:
    for h in hubs:
        expected_arcs.append((s, h))
for h in hubs:
    for c in customers:
        expected_arcs.append((h, c))
missing_arcs = [arc for arc in expected_arcs if arc not in arc_cost]
if missing_arcs:
    raise ValueError(f'Missing arc costs for arcs: {missing_arcs}')
m = gp.Model('DistributionNetwork')
f_vars = m.addVars(arc_cost.keys(), lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((arc_cost[arc] * f_vars[arc] for arc in arc_cost)), GRB.MINIMIZE)
for s in sources:
    m.addConstr(gp.quicksum((f_vars[s, h] for h in hubs)) <= source_supply[s], name=f'supply_{s}')
for c in customers:
    m.addConstr(gp.quicksum((f_vars[h, c] for h in hubs)) >= customer_demand[c], name=f'demand_{c}')
for h in hubs:
    m.addConstr(gp.quicksum((f_vars[s, h] for s in sources)) == gp.quicksum((f_vars[h, c] for c in customers)), name=f'flowbal_{h}')
for h in hubs:
    m.addConstr(gp.quicksum((f_vars[s, h] for s in sources)) <= hub_capacity[h], name=f'hubcap_{h}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')