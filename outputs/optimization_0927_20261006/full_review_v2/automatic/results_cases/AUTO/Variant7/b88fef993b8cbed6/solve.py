import gurobipy as gp
from gurobipy import GRB
sources = ['S1', 'S2', 'S3']
hubs = ['H1', 'H2']
customers = ['C1', 'C2', 'C3', 'C4']
source_supply = {'S1': 120, 'S2': 100, 'S3': 90}
customer_demand = {'C1': 70, 'C2': 80, 'C3': 60, 'C4': 90}
hub_capacity = {'H1': 170, 'H2': 160}
arc_cost = {('S1', 'H1'): 2, ('S1', 'H2'): 6, ('S2', 'H1'): 4, ('S2', 'H2'): 3, ('S3', 'H1'): 7, ('S3', 'H2'): 2, ('H1', 'C1'): 3, ('H1', 'C2'): 4, ('H1', 'C3'): 7, ('H1', 'C4'): 8, ('H2', 'C1'): 8, ('H2', 'C2'): 6, ('H2', 'C3'): 3, ('H2', 'C4'): 4}
source_hub_arcs = [(s, h) for s in sources for h in hubs]
hub_customer_arcs = [(h, c) for h in hubs for c in customers]
all_arcs = source_hub_arcs + hub_customer_arcs
missing_arcs = [arc for arc in all_arcs if arc not in arc_cost]
if missing_arcs:
    raise ValueError(f'Missing arc costs for arcs: {missing_arcs}')

def build_model():
    m = gp.Model('min_cost_transshipment')
    m.Params.MIPGap = 0.0001
    flow_vars = m.addVars(all_arcs, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((arc_cost[arc] * flow_vars[arc] for arc in all_arcs)), GRB.MINIMIZE)
    for s in sources:
        m.addConstr(gp.quicksum((flow_vars[s, h] for h in hubs)) <= source_supply[s], name=f'supply_{s}')
    for c in customers:
        m.addConstr(gp.quicksum((flow_vars[h, c] for h in hubs)) >= customer_demand[c], name=f'demand_{c}')
    for h in hubs:
        m.addConstr(gp.quicksum((flow_vars[s, h] for s in sources)) == gp.quicksum((flow_vars[h, c] for c in customers)), name=f'flowbal_{h}')
    for h in hubs:
        m.addConstr(gp.quicksum((flow_vars[s, h] for s in sources)) <= hub_capacity[h], name=f'hubcap_{h}')
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')