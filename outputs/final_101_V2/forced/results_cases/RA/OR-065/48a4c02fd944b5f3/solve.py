LEGACY_OBSERVATION = 'fixed_cost.csv\n{"values": {"Unnamed: 0": "S1", "fixed_costs": "102.33"}}\n{"values": {"Unnamed: 0": "S2", "fixed_costs": "94.92"}}\n{"values": {"Unnamed: 0": "S3", "fixed_costs": "91.83"}}\n\ntransportation_costs.csv\n{"values": {"Unnamed: 0": "S1", "C1": "1506.22", "C2": "70.90000000000001", "C3": "8.44"}}\n{"values": {"Unnamed: 0": "S2", "C1": "1732.65", "C2": "1780.72", "C3": "567.4400000000001"}}\n{"values": {"Unnamed: 0": "S3", "C1": "115.66", "C2": "100.76", "C3": "64.68000000000001"}}\n\ndemand.csv\n{"values": {"customer": "C1", "demand": "1083"}}\n{"values": {"customer": "C2", "demand": "776"}}\n{"values": {"customer": "C3", "demand": "16214"}}'
LEGACY_RECORDS = [{'source': 'fixed_cost.csv', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}}, {'source': 'fixed_cost.csv', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}}, {'source': 'fixed_cost.csv', 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001'}}, {'source': 'demand.csv', 'values': {'customer': 'C1', 'demand': '1083'}}, {'source': 'demand.csv', 'values': {'customer': 'C2', 'demand': '776'}}, {'source': 'demand.csv', 'values': {'customer': 'C3', 'demand': '16214'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
warehouses = []
fixed_cost = {}
customers = []
transport_cost = {}
demand = {}
for rec in records:
    if rec['source'] == 'fixed_cost.csv':
        wh = rec['values']['Unnamed: 0']
        warehouses.append(wh)
        fixed_cost[wh] = float(rec['values']['fixed_costs'])
for rec in records:
    if rec['source'] == 'transportation_costs.csv':
        wh = rec['values']['Unnamed: 0']
        if wh not in transport_cost:
            transport_cost[wh] = {}
        for k, v in rec['values'].items():
            if k == 'Unnamed: 0':
                continue
            if k not in customers:
                customers.append(k)
            transport_cost[wh][k] = float(v)
for rec in records:
    if rec['source'] == 'demand.csv':
        cust = rec['values']['customer']
        demand[cust] = float(rec['values']['demand'])
for wh in warehouses:
    if wh not in fixed_cost:
        raise ValueError(f'Missing fixed cost for warehouse {wh}')
    if wh not in transport_cost:
        raise ValueError(f'Missing transport cost row for warehouse {wh}')
    for cust in customers:
        if cust not in transport_cost[wh]:
            raise ValueError(f'Missing transport cost for warehouse {wh}, customer {cust}')
for cust in customers:
    if cust not in demand:
        raise ValueError(f'Missing demand for customer {cust}')
m = gp.Model('Bandcamp_Warehouse_Selection')
y = m.addVars(warehouses, vtype=GRB.BINARY, name='')
x = m.addVars(warehouses, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((fixed_cost[wh] * y[wh] for wh in warehouses)) + gp.quicksum((transport_cost[wh][cust] * x[wh, cust] for wh in warehouses for cust in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[wh, cust] for wh in warehouses)) == demand[cust] for cust in customers), name='')
m.addConstrs((x[wh, cust] <= demand[cust] * y[wh] for wh in warehouses for cust in customers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')