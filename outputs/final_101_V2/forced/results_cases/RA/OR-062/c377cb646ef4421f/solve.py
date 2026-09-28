LEGACY_OBSERVATION = 'fixed_cost.csv\n{"values": {"Unnamed: 0": "MOUNT AYR", "fixed_costs": "96.58"}}\n{"values": {"Unnamed: 0": "WAUKEE", "fixed_costs": "94.06"}}\n{"values": {"Unnamed: 0": "WAVERLY", "fixed_costs": "94.37"}}\n{"values": {"Unnamed: 0": "PELLA", "fixed_costs": "82.88"}}\n{"values": {"Unnamed: 0": "DES MOINES", "fixed_costs": "94.95999999999999"}}\n\ntransportation_costs.csv\n{"values": {"Unnamed: 0": "MOUNT AYR", "CLARINDA": "694.6799999999999", "FORT MADISON": "17.48", "SIOUX CITY": "20.07", "TOLEDO": "199.02", "BANCROFT": "1685.53"}}\n{"values": {"Unnamed: 0": "WAUKEE", "CLARINDA": "15.13", "FORT MADISON": "1.5", "SIOUX CITY": "1.43", "TOLEDO": "27.88", "BANCROFT": "90.69"}}\n{"values": {"Unnamed: 0": "WAVERLY", "CLARINDA": "2.34", "FORT MADISON": "349.34", "SIOUX CITY": "246.6", "TOLEDO": "41.3", "BANCROFT": "78.73"}}\n{"values": {"Unnamed: 0": "PELLA", "CLARINDA": "1181.6", "FORT MADISON": "1458.53", "SIOUX CITY": "1646.36", "TOLEDO": "1924.55", "BANCROFT": "38.93"}}\n{"values": {"Unnamed: 0": "DES MOINES", "CLARINDA": "1030.8", "FORT MADISON": "43.48", "SIOUX CITY": "932.4299999999999", "TOLEDO": "55.39", "BANCROFT": "103.84"}}\n\ndemand.csv\n{"values": {"Customer": "Customer_1", "demand": "2397"}}\n{"values": {"Customer": "Customer_2", "demand": "1889"}}\n{"values": {"Customer": "Customer_3", "demand": "2518"}}\n{"values": {"Customer": "Customer_4", "demand": "3218"}}\n{"values": {"Customer": "Customer_5", "demand": "1813"}}'
LEGACY_RECORDS = [{'source': 'fixed_cost.csv', 'values': {'Unnamed: 0': 'MOUNT AYR', 'fixed_costs': '96.58'}}, {'source': 'fixed_cost.csv', 'values': {'Unnamed: 0': 'WAUKEE', 'fixed_costs': '94.06'}}, {'source': 'fixed_cost.csv', 'values': {'Unnamed: 0': 'WAVERLY', 'fixed_costs': '94.37'}}, {'source': 'fixed_cost.csv', 'values': {'Unnamed: 0': 'PELLA', 'fixed_costs': '82.88'}}, {'source': 'fixed_cost.csv', 'values': {'Unnamed: 0': 'DES MOINES', 'fixed_costs': '94.95999999999999'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'MOUNT AYR', 'CLARINDA': '694.6799999999999', 'FORT MADISON': '17.48', 'SIOUX CITY': '20.07', 'TOLEDO': '199.02', 'BANCROFT': '1685.53'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'WAUKEE', 'CLARINDA': '15.13', 'FORT MADISON': '1.5', 'SIOUX CITY': '1.43', 'TOLEDO': '27.88', 'BANCROFT': '90.69'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'WAVERLY', 'CLARINDA': '2.34', 'FORT MADISON': '349.34', 'SIOUX CITY': '246.6', 'TOLEDO': '41.3', 'BANCROFT': '78.73'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'PELLA', 'CLARINDA': '1181.6', 'FORT MADISON': '1458.53', 'SIOUX CITY': '1646.36', 'TOLEDO': '1924.55', 'BANCROFT': '38.93'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'DES MOINES', 'CLARINDA': '1030.8', 'FORT MADISON': '43.48', 'SIOUX CITY': '932.4299999999999', 'TOLEDO': '55.39', 'BANCROFT': '103.84'}}, {'source': 'demand.csv', 'values': {'Customer': 'Customer_1', 'demand': '2397'}}, {'source': 'demand.csv', 'values': {'Customer': 'Customer_2', 'demand': '1889'}}, {'source': 'demand.csv', 'values': {'Customer': 'Customer_3', 'demand': '2518'}}, {'source': 'demand.csv', 'values': {'Customer': 'Customer_4', 'demand': '3218'}}, {'source': 'demand.csv', 'values': {'Customer': 'Customer_5', 'demand': '1813'}}]
import gurobipy as gp
from gurobipy import GRB
suppliers = []
fixed_costs = {}
stores = []
transportation_costs = {}
customers = []
demands = []
for rec in LEGACY_RECORDS:
    if rec['source'] == 'fixed_cost.csv':
        name = rec['values']['Unnamed: 0']
        suppliers.append(name)
        fixed_costs[name] = float(rec['values']['fixed_costs'])
for rec in LEGACY_RECORDS:
    if rec['source'] == 'transportation_costs.csv':
        supplier = rec['values']['Unnamed: 0']
        if not stores:
            stores = [k for k in rec['values'].keys() if k != 'Unnamed: 0']
        transportation_costs[supplier] = {}
        for store in stores:
            transportation_costs[supplier][store] = float(rec['values'][store])
for rec in LEGACY_RECORDS:
    if rec['source'] == 'demand.csv':
        customers.append(rec['values']['Customer'])
        demands.append(int(rec['values']['demand']))
if len(customers) != len(stores):
    raise ValueError('Number of customers and stores do not match.')
customer_store_map = dict(zip(customers, stores))
store_demand = dict(zip(stores, demands))
M = sum(demands)
for s in suppliers:
    if s not in fixed_costs:
        raise ValueError(f'Missing fixed cost for supplier {s}')
    if s not in transportation_costs:
        raise ValueError(f'Missing transportation costs for supplier {s}')
    for j in stores:
        if j not in transportation_costs[s]:
            raise ValueError(f'Missing transportation cost for supplier {s} to store {j}')
for j in stores:
    if j not in store_demand:
        raise ValueError(f'Missing demand for store {j}')
m = gp.Model('Iowa_Liquor_Supplier_Selection')
x = m.addVars(suppliers, stores, lb=0, vtype=GRB.INTEGER, name='')
y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_costs[i] * y[i] for i in suppliers)) + gp.quicksum((transportation_costs[i][j] * x[i, j] for i in suppliers for j in stores)), GRB.MINIMIZE)
for idx, store in enumerate(stores):
    m.addConstr(gp.quicksum((x[i, store] for i in suppliers)) >= store_demand[store], name=f'demand_{store}')
for i in suppliers:
    m.addConstr(gp.quicksum((x[i, j] for j in stores)) <= M * y[i], name=f'supplier_{i}_activation')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')