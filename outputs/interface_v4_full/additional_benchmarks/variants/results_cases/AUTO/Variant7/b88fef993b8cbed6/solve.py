import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    sources = ['S1', 'S2', 'S3']
    hubs = ['H1', 'H2']
    customers = ['C1', 'C2', 'C3', 'C4']
    source_supply = {'S1': 120, 'S2': 100, 'S3': 90}
    customer_demand = {'C1': 70, 'C2': 80, 'C3': 60, 'C4': 90}
    hub_capacity = {'H1': 170, 'H2': 160}
    arc_costs = {('S1', 'H1'): 2, ('S1', 'H2'): 6, ('S2', 'H1'): 4, ('S2', 'H2'): 3, ('S3', 'H1'): 7, ('S3', 'H2'): 2, ('H1', 'C1'): 3, ('H1', 'C2'): 4, ('H1', 'C3'): 7, ('H1', 'C4'): 8, ('H2', 'C1'): 8, ('H2', 'C2'): 6, ('H2', 'C3'): 3, ('H2', 'C4'): 4}
    for s in sources:
        for h in hubs:
            if (s, h) not in arc_costs:
                raise ValueError(f'Missing arc cost for ({s},{h})')
    for h in hubs:
        for c in customers:
            if (h, c) not in arc_costs:
                raise ValueError(f'Missing arc cost for ({h},{c})')
    m = gp.Model('DistributionNetwork')
    arcs = list(arc_costs.keys())
    f = m.addVars(arcs, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((arc_costs[i_j] * f[i_j] for i_j in arcs)), GRB.MINIMIZE)
    for s in sources:
        m.addConstr(gp.quicksum((f[s, h] for h in hubs)) <= source_supply[s], name=f'supply_{s}')
    for c in customers:
        m.addConstr(gp.quicksum((f[h, c] for h in hubs)) >= customer_demand[c], name=f'demand_{c}')
    for h in hubs:
        m.addConstr(gp.quicksum((f[s, h] for s in sources)) == gp.quicksum((f[h, c] for c in customers)), name=f'flowbal_{h}')
    for h in hubs:
        m.addConstr(gp.quicksum((f[s, h] for s in sources)) <= hub_capacity[h], name=f'hubcap_{h}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()