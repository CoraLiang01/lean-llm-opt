import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    I = ['MOUNT AYR', 'WAUKEE', 'WAVERLY', 'PELLA', 'DES MOINES']
    J = ['Customer_1 (CLARINDA)', 'Customer_2 (FORT MADISON)', 'Customer_3 (SIOUX CITY)', 'Customer_4 (TOLEDO)', 'Customer_5 (BANCROFT)']
    f = {'MOUNT AYR': 96.58, 'WAUKEE': 94.06, 'WAVERLY': 94.37, 'PELLA': 82.88, 'DES MOINES': 94.96}
    c = {('MOUNT AYR', 'Customer_1 (CLARINDA)'): 694.68, ('MOUNT AYR', 'Customer_2 (FORT MADISON)'): 17.48, ('MOUNT AYR', 'Customer_3 (SIOUX CITY)'): 20.07, ('MOUNT AYR', 'Customer_4 (TOLEDO)'): 199.02, ('MOUNT AYR', 'Customer_5 (BANCROFT)'): 1685.53, ('WAUKEE', 'Customer_1 (CLARINDA)'): 15.13, ('WAUKEE', 'Customer_2 (FORT MADISON)'): 1.5, ('WAUKEE', 'Customer_3 (SIOUX CITY)'): 1.43, ('WAUKEE', 'Customer_4 (TOLEDO)'): 27.88, ('WAUKEE', 'Customer_5 (BANCROFT)'): 90.69, ('WAVERLY', 'Customer_1 (CLARINDA)'): 2.34, ('WAVERLY', 'Customer_2 (FORT MADISON)'): 349.34, ('WAVERLY', 'Customer_3 (SIOUX CITY)'): 246.6, ('WAVERLY', 'Customer_4 (TOLEDO)'): 41.3, ('WAVERLY', 'Customer_5 (BANCROFT)'): 78.73, ('PELLA', 'Customer_1 (CLARINDA)'): 1181.6, ('PELLA', 'Customer_2 (FORT MADISON)'): 1458.53, ('PELLA', 'Customer_3 (SIOUX CITY)'): 1646.36, ('PELLA', 'Customer_4 (TOLEDO)'): 1924.55, ('PELLA', 'Customer_5 (BANCROFT)'): 38.93, ('DES MOINES', 'Customer_1 (CLARINDA)'): 1030.8, ('DES MOINES', 'Customer_2 (FORT MADISON)'): 43.48, ('DES MOINES', 'Customer_3 (SIOUX CITY)'): 932.43, ('DES MOINES', 'Customer_4 (TOLEDO)'): 55.39, ('DES MOINES', 'Customer_5 (BANCROFT)'): 103.84}
    d = {'Customer_1 (CLARINDA)': 2397, 'Customer_2 (FORT MADISON)': 1889, 'Customer_3 (SIOUX CITY)': 2518, 'Customer_4 (TOLEDO)': 3218, 'Customer_5 (BANCROFT)': 1813}
    for i in I:
        if i not in f:
            raise ValueError(f'Missing fixed cost for supplier {i}')
    for j in J:
        if j not in d:
            raise ValueError(f'Missing demand for customer {j}')
    for i in I:
        for j in J:
            if (i, j) not in c:
                raise ValueError(f'Missing transportation cost for ({i}, {j})')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    y = m.addVars(I, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x = m.addVars(I, J, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((f[i] * y[i] for i in I)) + gp.quicksum((c[i, j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((x[i, j] for i in I)) == d[j], name='demand_' + str(j))
    for i in I:
        for j in J:
            m.addConstr(x[i, j] <= d[j] * y[i], name='link_' + str(i) + '_' + str(j))
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()