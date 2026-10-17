SOURCE_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv', 'values': {'Customer': 'Customer_1', 'demand': '2397'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv', 'values': {'Customer': 'Customer_2', 'demand': '1889'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv', 'values': {'Customer': 'Customer_3', 'demand': '2518'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv', 'values': {'Customer': 'Customer_4', 'demand': '3218'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv', 'values': {'Customer': 'Customer_5', 'demand': '1813'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv', 'values': {'Unnamed: 0': 'MOUNT AYR', 'fixed_costs': '96.58'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv', 'values': {'Unnamed: 0': 'WAUKEE', 'fixed_costs': '94.06'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv', 'values': {'Unnamed: 0': 'WAVERLY', 'fixed_costs': '94.37'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv', 'values': {'Unnamed: 0': 'PELLA', 'fixed_costs': '82.88'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv', 'values': {'Unnamed: 0': 'DES MOINES', 'fixed_costs': '94.95999999999999'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv', 'values': {'Unnamed: 0': 'MOUNT AYR', 'CLARINDA': '694.6799999999999', 'FORT MADISON': '17.48', 'SIOUX CITY': '20.07', 'TOLEDO': '199.02', 'BANCROFT': '1685.53'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv', 'values': {'Unnamed: 0': 'WAUKEE', 'CLARINDA': '15.13', 'FORT MADISON': '1.5', 'SIOUX CITY': '1.43', 'TOLEDO': '27.88', 'BANCROFT': '90.69'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv', 'values': {'Unnamed: 0': 'WAVERLY', 'CLARINDA': '2.34', 'FORT MADISON': '349.34', 'SIOUX CITY': '246.6', 'TOLEDO': '41.3', 'BANCROFT': '78.73'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv', 'values': {'Unnamed: 0': 'PELLA', 'CLARINDA': '1181.6', 'FORT MADISON': '1458.53', 'SIOUX CITY': '1646.36', 'TOLEDO': '1924.55', 'BANCROFT': '38.93'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv', 'values': {'Unnamed: 0': 'DES MOINES', 'CLARINDA': '1030.8', 'FORT MADISON': '43.48', 'SIOUX CITY': '932.4299999999999', 'TOLEDO': '55.39', 'BANCROFT': '103.84'}}]
import gurobipy as gp
from gurobipy import GRB
records = SOURCE_RECORDS
suppliers = []
stores = []
demand = {}
fixed_cost = {}
cost = {}
for rec in records:
    if rec['source'].endswith('demand.csv'):
        store = rec['values']['Customer']
        val = float(rec['values']['demand'])
        stores.append(store)
        demand[store] = val
for rec in records:
    if rec['source'].endswith('fixed_cost.csv'):
        supplier = rec['values']['Unnamed: 0']
        val = float(rec['values']['fixed_costs'])
        suppliers.append(supplier)
        fixed_cost[supplier] = val
for rec in records:
    if rec['source'].endswith('transportation_costs.csv'):
        supplier = rec['values']['Unnamed: 0']
        if supplier not in cost:
            cost[supplier] = {}
        for store in stores:
            if store in rec['values']:
                cost[supplier][store] = float(rec['values'][store])
if set(demand.keys()) != set(stores):
    raise ValueError('Demand keys do not match stores')
if set(fixed_cost.keys()) != set(suppliers):
    raise ValueError('Fixed cost keys do not match suppliers')
for i in suppliers:
    if i not in cost:
        raise ValueError(f'Missing cost row for supplier {i}')
    for j in stores:
        if j not in cost[i]:
            raise ValueError(f'Missing cost for supplier {i}, store {j}')
M = sum((demand[j] for j in stores))
m = gp.Model('Original_RAG_FLP')
x_vars = m.addVars(suppliers, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in suppliers for j in stores)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in suppliers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in suppliers)) == demand[j] for j in stores), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in stores)) <= M * y_vars[i] for i in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')