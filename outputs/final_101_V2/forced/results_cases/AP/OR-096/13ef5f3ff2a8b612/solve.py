import gurobipy as gp
from gurobipy import GRB
schools = ['I', 'II']
neighborhoods = ['N01', 'N02', 'N03', 'N04', 'N05', 'N06', 'N07', 'N08', 'N09', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'N17', 'N18', 'N19', 'N20', 'N21', 'N22', 'N23', 'N24', 'N25', 'N26', 'N27', 'N28', 'N29', 'N30', 'N31']
races = ['White', 'NonWhite']
school_capacity = {'I': 2028, 'II': 1560}
neighborhoods_population = {'N01': {'White': 78, 'NonWhite': 22}, 'N02': {'White': 57, 'NonWhite': 33}, 'N03': {'White': 47, 'NonWhite': 63}, 'N04': {'White': 78, 'NonWhite': 22}, 'N05': {'White': 57, 'NonWhite': 33}, 'N06': {'White': 47, 'NonWhite': 63}, 'N07': {'White': 78, 'NonWhite': 22}, 'N08': {'White': 57, 'NonWhite': 33}, 'N09': {'White': 46, 'NonWhite': 64}, 'N10': {'White': 77, 'NonWhite': 23}, 'N11': {'White': 56, 'NonWhite': 34}, 'N12': {'White': 46, 'NonWhite': 64}, 'N13': {'White': 77, 'NonWhite': 23}, 'N14': {'White': 56, 'NonWhite': 34}, 'N15': {'White': 46, 'NonWhite': 64}, 'N16': {'White': 77, 'NonWhite': 23}, 'N17': {'White': 56, 'NonWhite': 34}, 'N18': {'White': 46, 'NonWhite': 64}, 'N19': {'White': 77, 'NonWhite': 23}, 'N20': {'White': 56, 'NonWhite': 34}, 'N21': {'White': 46, 'NonWhite': 64}, 'N22': {'White': 77, 'NonWhite': 23}, 'N23': {'White': 56, 'NonWhite': 34}, 'N24': {'White': 46, 'NonWhite': 64}, 'N25': {'White': 77, 'NonWhite': 23}, 'N26': {'White': 56, 'NonWhite': 34}, 'N27': {'White': 46, 'NonWhite': 64}, 'N28': {'White': 77, 'NonWhite': 23}, 'N29': {'White': 56, 'NonWhite': 34}, 'N30': {'White': 46, 'NonWhite': 64}, 'N31': {'White': 74, 'NonWhite': 46}}
distance = {'I': {'N01': 1.25, 'N02': 1.3, 'N03': 1.35, 'N04': 1.4, 'N05': 1.45, 'N06': 1.5, 'N07': 1.55, 'N08': 1.6, 'N09': 1.65, 'N10': 1.7, 'N11': 1.75, 'N12': 1.8, 'N13': 1.85, 'N14': 1.9, 'N15': 1.95, 'N16': 2.0, 'N17': 3.08, 'N18': 3.16, 'N19': 3.24, 'N20': 3.32, 'N21': 3.4, 'N22': 3.48, 'N23': 3.56, 'N24': 3.64, 'N25': 3.72, 'N26': 3.8, 'N27': 3.88, 'N28': 3.96, 'N29': 4.04, 'N30': 4.12, 'N31': 4.2}, 'II': {'N01': 3.08, 'N02': 3.16, 'N03': 3.24, 'N04': 3.32, 'N05': 3.4, 'N06': 3.48, 'N07': 3.56, 'N08': 3.64, 'N09': 3.72, 'N10': 3.8, 'N11': 3.88, 'N12': 3.96, 'N13': 4.04, 'N14': 4.12, 'N15': 4.2, 'N16': 4.28, 'N17': 1.25, 'N18': 1.3, 'N19': 1.35, 'N20': 1.4, 'N21': 1.45, 'N22': 1.5, 'N23': 1.55, 'N24': 1.6, 'N25': 1.65, 'N26': 1.7, 'N27': 1.75, 'N28': 1.8, 'N29': 1.85, 'N30': 1.9, 'N31': 1.95}}
for s in schools:
    if s not in school_capacity:
        raise ValueError(f'Missing capacity for school {s}')
    if s not in distance:
        raise ValueError(f'Missing distance data for school {s}')
    for n in neighborhoods:
        if n not in distance[s]:
            raise ValueError(f'Missing distance for school {s}, neighborhood {n}')
for n in neighborhoods:
    if n not in neighborhoods_population:
        raise ValueError(f'Missing population for neighborhood {n}')
    for r in races:
        if r not in neighborhoods_population[n]:
            raise ValueError(f'Missing population for neighborhood {n}, race {r}')
m = gp.Model('SchoolAssignment')
x = m.addVars(schools, neighborhoods, races, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((distance[s][n] * x[s, n, r] for s in schools for n in neighborhoods for r in races)), GRB.MINIMIZE)
for n in neighborhoods:
    for r in races:
        m.addConstr(gp.quicksum((x[s, n, r] for s in schools)) == neighborhoods_population[n][r], name='pop')
for s in schools:
    m.addConstr(gp.quicksum((x[s, n, r] for n in neighborhoods for r in races)) <= school_capacity[s], name='cap')
for s in schools:
    T_s = gp.quicksum((x[s, n, r] for n in neighborhoods for r in races))
    W_s = gp.quicksum((x[s, n, 'White'] for n in neighborhoods))
    m.addConstr(W_s >= 0.5 * T_s, name='rb_lo')
    m.addConstr(W_s <= 0.7 * T_s, name='rb_hi')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')