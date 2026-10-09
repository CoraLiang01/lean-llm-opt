import gurobipy as gp
from gurobipy import GRB

def solve_greenmart_transportation():
    warehouses = ['S1', 'S2', 'S3', 'S4', 'S5']
    stores = ['D1', 'D2', 'D3', 'D4', 'D5']
    customer_demand = {'D1': 428, 'D2': 217, 'D3': 214, 'D4': 380, 'D5': 254}
    supply_capacity = {'S1': 428, 'S2': 217, 'S3': 214, 'S4': 380, 'S5': 254}
    transportation_costs = {'S1': {'D1': 269.3910588020795, 'D2': 1.4537335390933939, 'D3': 99.60345345756605, 'D4': 26.64078166309837, 'D5': 9.537688956880922}, 'S2': {'D1': 9.291846876785183, 'D2': 10.874778437070223, 'D3': 144.52609291614627, 'D4': 11.420133077898234, 'D5': 153.1756819927813}, 'S3': {'D1': 9.674584301671008, 'D2': 2.6191650959687944, 'D3': 100.8242249168735, 'D4': 3.2121910887916876, 'D5': 133.8493396124168}, 'S4': {'D1': 270.57498480010247, 'D2': 32.50253586, 'D3': 4.6842098096469815, 'D4': 1.5682269686546804, 'D5': 9.58927599}, 'S5': {'D1': 226.0331910675782, 'D2': 8.669161980826471, 'D3': 65.47681316968448, 'D4': 9.068765258459958, 'D5': 202.65015316425533}}
    for s in warehouses:
        if s not in transportation_costs:
            raise ValueError(f'Missing transportation costs for warehouse {s}')
        for d in stores:
            if d not in transportation_costs[s]:
                raise ValueError(f'Missing transportation cost for warehouse {s} to store {d}')
    for d in stores:
        if d not in customer_demand:
            raise ValueError(f'Missing demand for store {d}')
    for s in warehouses:
        if s not in supply_capacity:
            raise ValueError(f'Missing supply capacity for warehouse {s}')
    m = gp.Model()
    x = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = gp.LinExpr()
    for s in warehouses:
        for d in stores:
            obj += transportation_costs[s][d] * x[s, d]
    m.setObjective(obj, GRB.MINIMIZE)
    for d in stores:
        m.addConstr(gp.quicksum((x[s, d] for s in warehouses)) == customer_demand[d], name='')
    for s in warehouses:
        m.addConstr(gp.quicksum((x[s, d] for d in stores)) <= supply_capacity[s], name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for s in warehouses:
            for d in stores:
                var = x[s, d]
                print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_greenmart_transportation()