import gurobipy as gp
from gurobipy import GRB
suppliers = ['MOUNT AYR', 'WAUKEE', 'WAVERLY', 'PELLA', 'DES MOINES']
customers = ['Customer_1', 'Customer_2', 'Customer_3', 'Customer_4', 'Customer_5']
customer_cities = ['CLARINDA', 'FORT MADISON', 'SIOUX CITY', 'TOLEDO', 'BANCROFT']
customer_to_city = {'Customer_1': 'CLARINDA', 'Customer_2': 'FORT MADISON', 'Customer_3': 'SIOUX CITY', 'Customer_4': 'TOLEDO', 'Customer_5': 'BANCROFT'}
demand = {'Customer_1': 2397, 'Customer_2': 1889, 'Customer_3': 2518, 'Customer_4': 3218, 'Customer_5': 1813}
fixed_cost = {'MOUNT AYR': 96.58, 'WAUKEE': 94.06, 'WAVERLY': 94.37, 'PELLA': 82.88, 'DES MOINES': 94.96}
cost = {'MOUNT AYR': {'CLARINDA': 694.68, 'FORT MADISON': 17.48, 'SIOUX CITY': 20.07, 'TOLEDO': 199.02, 'BANCROFT': 1685.53}, 'WAUKEE': {'CLARINDA': 15.13, 'FORT MADISON': 1.5, 'SIOUX CITY': 1.43, 'TOLEDO': 27.88, 'BANCROFT': 90.69}, 'WAVERLY': {'CLARINDA': 2.34, 'FORT MADISON': 349.34, 'SIOUX CITY': 246.6, 'TOLEDO': 41.3, 'BANCROFT': 78.73}, 'PELLA': {'CLARINDA': 1181.6, 'FORT MADISON': 1458.53, 'SIOUX CITY': 1646.36, 'TOLEDO': 1924.55, 'BANCROFT': 38.93}, 'DES MOINES': {'CLARINDA': 1030.8, 'FORT MADISON': 43.48, 'SIOUX CITY': 932.43, 'TOLEDO': 55.39, 'BANCROFT': 103.84}}
M = 11835
for i in suppliers:
    if i not in fixed_cost:
        raise ValueError(f'Missing fixed cost for supplier {i}')
    if i not in cost:
        raise ValueError(f'Missing cost row for supplier {i}')
    for jcity in customer_cities:
        if jcity not in cost[i]:
            raise ValueError(f'Missing cost for supplier {i} to city {jcity}')
for cust in customers:
    if cust not in demand:
        raise ValueError(f'Missing demand for customer {cust}')
    if cust not in customer_to_city:
        raise ValueError(f'Missing city mapping for customer {cust}')
    if customer_to_city[cust] not in customer_cities:
        raise ValueError(f'Customer {cust} mapped to unknown city {customer_to_city[cust]}')
m = gp.Model('Iowa_Liquor_FLP')
x_vars = m.addVars(suppliers, customer_cities, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][jcity] * x_vars[i, jcity] for i in suppliers for jcity in customer_cities)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in suppliers)), GRB.MINIMIZE)
for cust in customers:
    city = customer_to_city[cust]
    m.addConstr(gp.quicksum((x_vars[i, city] for i in suppliers)) == demand[cust], name=f'demand_{cust}')
for i in suppliers:
    m.addConstr(gp.quicksum((x_vars[i, jcity] for jcity in customer_cities)) <= M * y_vars[i], name=f'activation_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')