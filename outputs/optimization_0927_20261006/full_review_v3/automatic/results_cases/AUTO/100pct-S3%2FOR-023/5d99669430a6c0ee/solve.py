import gurobipy as gp
from gurobipy import GRB
suppliers = ['MOUNT AYR', 'WAUKEE', 'WAVERLY', 'PELLA', 'DES MOINES']
customers = ['Customer_1', 'Customer_2', 'Customer_3', 'Customer_4', 'Customer_5']
locations = ['CLARINDA', 'FORT MADISON', 'SIOUX CITY', 'TOLEDO', 'BANCROFT']
demand = {'Customer_1': 2397, 'Customer_2': 1889, 'Customer_3': 2518, 'Customer_4': 3218, 'Customer_5': 1813}
fixed_cost = {'MOUNT AYR': 96.58, 'WAUKEE': 94.06, 'WAVERLY': 94.37, 'PELLA': 82.88, 'DES MOINES': 94.96}
customer_location = {'Customer_1': 'CLARINDA', 'Customer_2': 'FORT MADISON', 'Customer_3': 'SIOUX CITY', 'Customer_4': 'TOLEDO', 'Customer_5': 'BANCROFT'}
transportation_cost_matrix = {'MOUNT AYR': {'CLARINDA': 694.68, 'FORT MADISON': 17.48, 'SIOUX CITY': 20.07, 'TOLEDO': 199.02, 'BANCROFT': 1685.53}, 'WAUKEE': {'CLARINDA': 15.13, 'FORT MADISON': 1.5, 'SIOUX CITY': 1.43, 'TOLEDO': 27.88, 'BANCROFT': 90.69}, 'WAVERLY': {'CLARINDA': 2.34, 'FORT MADISON': 349.34, 'SIOUX CITY': 246.6, 'TOLEDO': 41.3, 'BANCROFT': 78.73}, 'PELLA': {'CLARINDA': 1181.6, 'FORT MADISON': 1458.53, 'SIOUX CITY': 1646.36, 'TOLEDO': 1924.55, 'BANCROFT': 38.93}, 'DES MOINES': {'CLARINDA': 1030.8, 'FORT MADISON': 43.48, 'SIOUX CITY': 932.43, 'TOLEDO': 55.39, 'BANCROFT': 103.84}}
cost = {}
for i in suppliers:
    cost[i] = {}
    for j in customers:
        loc = customer_location[j]
        if loc not in transportation_cost_matrix[i]:
            raise ValueError(f'Missing transportation cost for supplier {i} to location {loc}')
        cost[i][j] = transportation_cost_matrix[i][loc]
for i in suppliers:
    if i not in fixed_cost:
        raise ValueError(f'Missing fixed cost for supplier {i}')
    for j in customers:
        if j not in cost[i]:
            raise ValueError(f'Missing cost for supplier {i} to customer {j}')
for j in customers:
    if j not in demand:
        raise ValueError(f'Missing demand for customer {j}')
M = sum(demand.values())

def build_model():
    m = gp.Model('Iowa_Liquor_FLP')
    x_vars = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in suppliers for j in customers)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x_vars[i, j] for j in customers)) <= M * y_vars[i] for i in suppliers), name='')
    m.Params.MIPGap = 0.0001
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')