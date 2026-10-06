import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    S = ['I', 'II']
    N = ['N01', 'N02', 'N03', 'N04', 'N05', 'N06', 'N07', 'N08', 'N09', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'N17', 'N18', 'N19', 'N20', 'N21', 'N22', 'N23', 'N24', 'N25', 'N26', 'N27', 'N28', 'N29', 'N30', 'N31']
    R = ['W', 'NW']
    C = {'I': 2028, 'II': 1560}
    P_W = {'N01': 78, 'N02': 57, 'N03': 47, 'N04': 78, 'N05': 57, 'N06': 47, 'N07': 78, 'N08': 57, 'N09': 46, 'N10': 77, 'N11': 56, 'N12': 46, 'N13': 77, 'N14': 56, 'N15': 46, 'N16': 77, 'N17': 56, 'N18': 46, 'N19': 77, 'N20': 56, 'N21': 46, 'N22': 77, 'N23': 56, 'N24': 46, 'N25': 77, 'N26': 56, 'N27': 46, 'N28': 77, 'N29': 56, 'N30': 46, 'N31': 74}
    P_NW = {'N01': 22, 'N02': 33, 'N03': 63, 'N04': 22, 'N05': 33, 'N06': 63, 'N07': 22, 'N08': 33, 'N09': 64, 'N10': 23, 'N11': 34, 'N12': 64, 'N13': 23, 'N14': 34, 'N15': 64, 'N16': 23, 'N17': 34, 'N18': 64, 'N19': 23, 'N20': 34, 'N21': 64, 'N22': 23, 'N23': 34, 'N24': 64, 'N25': 23, 'N26': 34, 'N27': 64, 'N28': 23, 'N29': 34, 'N30': 64, 'N31': 46}
    d = {'I': {'N01': 1.25, 'N02': 1.3, 'N03': 1.35, 'N04': 1.4, 'N05': 1.45, 'N06': 1.5, 'N07': 1.55, 'N08': 1.6, 'N09': 1.65, 'N10': 1.7, 'N11': 1.75, 'N12': 1.8, 'N13': 1.85, 'N14': 1.9, 'N15': 1.95, 'N16': 2.0, 'N17': 3.08, 'N18': 3.16, 'N19': 3.24, 'N20': 3.32, 'N21': 3.4, 'N22': 3.48, 'N23': 3.56, 'N24': 3.64, 'N25': 3.72, 'N26': 3.8, 'N27': 3.88, 'N28': 3.96, 'N29': 4.04, 'N30': 4.12, 'N31': 4.2}, 'II': {'N01': 3.08, 'N02': 3.16, 'N03': 3.24, 'N04': 3.32, 'N05': 3.4, 'N06': 3.48, 'N07': 3.56, 'N08': 3.64, 'N09': 3.72, 'N10': 3.8, 'N11': 3.88, 'N12': 3.96, 'N13': 4.04, 'N14': 4.12, 'N15': 4.2, 'N16': 4.28, 'N17': 1.25, 'N18': 1.3, 'N19': 1.35, 'N20': 1.4, 'N21': 1.45, 'N22': 1.5, 'N23': 1.55, 'N24': 1.6, 'N25': 1.65, 'N26': 1.7, 'N27': 1.75, 'N28': 1.8, 'N29': 1.85, 'N30': 1.9, 'N31': 1.95}}
    p_W = 0.6
    p_NW = 0.4
    dev = 0.1
    for s in S:
        if s not in d:
            raise ValueError(f'Missing distances for school {s}')
        for n in N:
            if n not in d[s]:
                raise ValueError(f'Missing distance for school {s}, neighborhood {n}')
    for n in N:
        if n not in P_W or n not in P_NW:
            raise ValueError(f'Missing population for neighborhood {n}')
    m = gp.Model('SchoolAssignment')
    m.Params.MIPGap = 0.0001
    x = m.addVars(S, N, R, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((d[s][n] * (x[s, n, 'W'] + x[s, n, 'NW']) for s in S for n in N)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[s, n, 'W'] for s in S)) == P_W[n] for n in N), name='')
    m.addConstrs((gp.quicksum((x[s, n, 'NW'] for s in S)) == P_NW[n] for n in N), name='')
    m.addConstrs((gp.quicksum((x[s, n, 'W'] + x[s, n, 'NW'] for n in N)) <= C[s] for s in S), name='')
    for s in S:
        total_white = gp.quicksum((x[s, n, 'W'] for n in N))
        total_students = gp.quicksum((x[s, n, 'W'] + x[s, n, 'NW'] for n in N))
        m.addConstr(total_white - (p_W + dev) * total_students <= 0, name=f'race_upper_{s}')
        m.addConstr(total_white - (p_W - dev) * total_students >= 0, name=f'race_lower_{s}')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')