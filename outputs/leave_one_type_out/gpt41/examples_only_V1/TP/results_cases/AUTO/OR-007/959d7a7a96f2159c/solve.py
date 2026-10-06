import gurobipy as gp
from gurobipy import GRB

def solve_greenmart_transportation():
    S = ['S1', 'S2', 'S3', 'S4', 'S5']
    D = ['D1', 'D2', 'D3', 'D4', 'D5']
    demand = {'D1': 428, 'D2': 217, 'D3': 214, 'D4': 380, 'D5': 254}
    supply = {'S1': 428, 'S2': 217, 'S3': 214, 'S4': 380, 'S5': 254}
    cost = {'S1': {'D1': 269.3910588020795, 'D2': 1.4537335390933939, 'D3': 99.60345345756605, 'D4': 26.64078166309837, 'D5': 9.537688956880922}, 'S2': {'D1': 9.291846876785183, 'D2': 10.874778437070223, 'D3': 144.52609291614627, 'D4': 11.420133077898234, 'D5': 153.1756819927813}, 'S3': {'D1': 9.674584301671008, 'D2': 2.6191650959687944, 'D3': 100.8242249168735, 'D4': 3.2121910887916876, 'D5': 133.8493396124168}, 'S4': {'D1': 270.57498480010247, 'D2': 32.50253586, 'D3': 4.6842098096469815, 'D4': 1.5682269686546804, 'D5': 9.58927599}, 'S5': {'D1': 226.0331910675782, 'D2': 8.669161980826471, 'D3': 65.47681316968448, 'D4': 9.068765258459958, 'D5': 202.65015316425533}}
    for i in S:
        if i not in cost:
            raise ValueError(f'Missing cost data for warehouse {i}')
        for j in D:
            if j not in cost[i]:
                raise ValueError(f'Missing cost data for warehouse {i} to store {j}')
    for j in D:
        if j not in demand:
            raise ValueError(f'Missing demand data for store {j}')
    for i in S:
        if i not in supply:
            raise ValueError(f'Missing supply data for warehouse {i}')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    x = {}
    for i in S:
        x[i] = {}
        for j in D:
            x[i][j] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='x')
    for j in D:
        m.addConstr(gp.quicksum((x[i][j] for i in S)) == demand[j], name='d')
    for i in S:
        m.addConstr(gp.quicksum((x[i][j] for j in D)) <= supply[i], name='s')
    obj = gp.LinExpr()
    for i in S:
        for j in D:
            obj += cost[i][j] * x[i][j]
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in S:
            for j in D:
                v = x[i][j]
                print(f'{v.VarName}[{i},{j}] {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_greenmart_transportation()