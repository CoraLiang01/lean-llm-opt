import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    facilities = [{'id': 1, 'name': 'MOUNT AYR', 'fixed_cost': 96.58}, {'id': 2, 'name': 'WAUKEE', 'fixed_cost': 94.06}, {'id': 3, 'name': 'WAVERLY', 'fixed_cost': 94.37}, {'id': 4, 'name': 'PELLA', 'fixed_cost': 82.88}, {'id': 5, 'name': 'DES MOINES', 'fixed_cost': 94.96}]
    customers = [{'id': 'Customer_1', 'demand': 2397}, {'id': 'Customer_2', 'demand': 1889}, {'id': 'Customer_3', 'demand': 2518}, {'id': 'Customer_4', 'demand': 3218}, {'id': 'Customer_5', 'demand': 1813}]
    transportation_costs = {'MOUNT AYR': {'Customer_1': 694.68, 'Customer_2': 17.48, 'Customer_3': 20.07, 'Customer_4': 199.02, 'Customer_5': 1685.53}, 'WAUKEE': {'Customer_1': 15.13, 'Customer_2': 1.5, 'Customer_3': 1.43, 'Customer_4': 27.88, 'Customer_5': 90.69}, 'WAVERLY': {'Customer_1': 2.34, 'Customer_2': 349.34, 'Customer_3': 246.6, 'Customer_4': 41.3, 'Customer_5': 78.73}, 'PELLA': {'Customer_1': 1181.6, 'Customer_2': 1458.53, 'Customer_3': 1646.36, 'Customer_4': 1924.55, 'Customer_5': 38.93}, 'DES MOINES': {'Customer_1': 1030.8, 'Customer_2': 43.48, 'Customer_3': 932.43, 'Customer_4': 55.39, 'Customer_5': 103.84}}
    M = 11835
    I = [f['name'] for f in facilities]
    J = [c['id'] for c in customers]
    f_i = {f['name']: f['fixed_cost'] for f in facilities}
    d_j = {c['id']: c['demand'] for c in customers}
    c_ij = transportation_costs
    for i in I:
        if i not in c_ij:
            raise ValueError(f'Missing transportation costs for facility {i}')
        for j in J:
            if j not in c_ij[i]:
                raise ValueError(f'Missing transportation cost for facility {i}, customer {j}')
    for i in I:
        if i not in f_i:
            raise ValueError(f'Missing fixed cost for facility {i}')
    for j in J:
        if j not in d_j:
            raise ValueError(f'Missing demand for customer {j}')
    m = gp.Model('Iowa_Liquor_Facility_Location')
    x = m.addVars(I, J, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c_ij[i][j] * x[i, j] for i in I for j in J)) + gp.quicksum((f_i[i] * y[i] for i in I)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in I)) == d_j[j] for j in J), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in J)) <= M * y[i] for i in I), name='')
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