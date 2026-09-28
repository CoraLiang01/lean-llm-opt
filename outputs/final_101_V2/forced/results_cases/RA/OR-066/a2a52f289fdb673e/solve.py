LEGACY_OBSERVATION = 'fixed_cost.csv\n{"Unnamed: 0": "S1", "fixed_costs": "105.97"}\n{"Unnamed: 0": "S2", "fixed_costs": "85.31"}\n\ntransportation_costs.csv\n{"Unnamed: 0": "S1", "C1": "2358.39", "C2": "1492.08"}\n{"Unnamed: 0": "S2", "C1": "0.07000000000000001", "C2": "52.32"}\n\ndemand.csv\n{"values": {"customer": "C1", "demand": "144"}}\n{"values": {"customer": "C2", "demand": "216"}}'
LEGACY_RECORDS = [{'source': 'fixed_cost.csv', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '105.97'}}, {'source': 'fixed_cost.csv', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '85.31'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '2358.39', 'C2': '1492.08'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '0.07000000000000001', 'C2': '52.32'}}, {'source': 'demand.csv', 'values': {'customer': 'C1', 'demand': '144'}}, {'source': 'demand.csv', 'values': {'customer': 'C2', 'demand': '216'}}]
import gurobipy as gp
from gurobipy import GRB
fixed_costs = {}
transport_costs = {}
demands = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'fixed_cost.csv':
        supplier = rec['values']['Unnamed: 0']
        fixed_costs[supplier] = float(rec['values']['fixed_costs'])
    elif rec['source'] == 'transportation_costs.csv':
        supplier = rec['values']['Unnamed: 0']
        transport_costs.setdefault(supplier, {})
        for k, v in rec['values'].items():
            if k != 'Unnamed: 0':
                transport_costs[supplier][k] = float(v)
    elif rec['source'] == 'demand.csv':
        customer = rec['values']['customer']
        demands[customer] = float(rec['values']['demand'])
suppliers = sorted(fixed_costs.keys())
customers = sorted(demands.keys())
for i in suppliers:
    if i not in fixed_costs:
        raise ValueError(f'Missing fixed cost for supplier {i}')
    if i not in transport_costs:
        raise ValueError(f'Missing transport costs for supplier {i}')
    for j in customers:
        if j not in transport_costs[i]:
            raise ValueError(f'Missing transport cost for supplier {i} to customer {j}')
for j in customers:
    if j not in demands:
        raise ValueError(f'Missing demand for customer {j}')
m = gp.Model('Supplier_Activation_Distribution')
y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((fixed_costs[i] * y[i] for i in suppliers)) + gp.quicksum((transport_costs[i][j] * x[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demands[j] for j in customers), name='')
m.addConstrs((x[i, j] <= demands[j] * y[i] for i in suppliers for j in customers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')