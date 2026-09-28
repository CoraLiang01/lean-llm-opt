LEGACY_OBSERVATION = '{"values": {"customer": "C1", "demand": "1083"}}\n{"values": {"customer": "C2", "demand": "776"}}\n{"values": {"customer": "C3", "demand": "16214"}}\n{"values": {"customer": "C4", "demand": "553"}}\n{"values": {"customer": "C5", "demand": "17106"}}\n{"values": {"customer": "C6", "demand": "594"}}\n{"values": {"customer": "C7", "demand": "732"}}\n{"values": {"Unnamed: 0": "S1", "fixed_costs": "102.33"}}\n{"values": {"Unnamed: 0": "S2", "fixed_costs": "94.92"}}\n{"values": {"Unnamed: 0": "S3", "fixed_costs": "91.83"}}\n{"values": {"Unnamed: 0": "S4", "fixed_costs": "98.70999999999999"}}\n{"values": {"Unnamed: 0": "S5", "fixed_costs": "95.73"}}\n{"values": {"Unnamed: 0": "S6", "fixed_costs": "99.95999999999999"}}\n{"values": {"Unnamed: 0": "S7", "fixed_costs": "98.16"}}\n{"values": {"Unnamed: 0": "S1", "C1": "1506.22", "C2": "70.90000000000001", "C3": "8.44", "C4": "260.27", "C5": "197.47", "C6": "71.70999999999999", "C7": "61.19"}}\n{"values": {"Unnamed: 0": "S2", "C1": "1732.65", "C2": "1780.72", "C3": "567.4400000000001", "C4": "448.68", "C5": "29", "C6": "1484.91", "C7": "963.92"}}\n{"values": {"Unnamed: 0": "S3", "C1": "115.66", "C2": "100.76", "C3": "64.68000000000001", "C4": "1324.53", "C5": "64.98999999999999", "C6": "134.88", "C7": "2102.83"}}\n{"values": {"Unnamed: 0": "S4", "C1": "1254.78", "C2": "1115.63", "C3": "52.31", "C4": "1036.16", "C5": "892.63", "C6": "1464.04", "C7": "1383.41"}}\n{"values": {"Unnamed: 0": "S5", "C1": "42.9", "C2": "891.01", "C3": "1013.94", "C4": "1128.72", "C5": "58.91", "C6": "42.89", "C7": "1570.31"}}\n{"values": {"Unnamed: 0": "S6", "C1": "0.7", "C2": "139.46", "C3": "70.03", "C4": "79.15000000000001", "C5": "1482", "C6": "0.91", "C7": "110.46"}}\n{"values": {"Unnamed: 0": "S7", "C1": "1732.3", "C2": "1780.44", "C3": "486.5", "C4": "523.74", "C5": "522.08", "C6": "82.48", "C7": "826.41"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'customer': 'C1', 'demand': '1083'}}, {'source': '', 'values': {'customer': 'C2', 'demand': '776'}}, {'source': '', 'values': {'customer': 'C3', 'demand': '16214'}}, {'source': '', 'values': {'customer': 'C4', 'demand': '553'}}, {'source': '', 'values': {'customer': 'C5', 'demand': '17106'}}, {'source': '', 'values': {'customer': 'C6', 'demand': '594'}}, {'source': '', 'values': {'customer': 'C7', 'demand': '732'}}, {'source': '', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}}, {'source': '', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}}, {'source': '', 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}, {'source': '', 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '98.70999999999999'}}, {'source': '', 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '95.73'}}, {'source': '', 'values': {'Unnamed: 0': 'S6', 'fixed_costs': '99.95999999999999'}}, {'source': '', 'values': {'Unnamed: 0': 'S7', 'fixed_costs': '98.16'}}, {'source': '', 'values': {'Unnamed: 0': 'S1', 'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44', 'C4': '260.27', 'C5': '197.47', 'C6': '71.70999999999999', 'C7': '61.19'}}, {'source': '', 'values': {'Unnamed: 0': 'S2', 'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001', 'C4': '448.68', 'C5': '29', 'C6': '1484.91', 'C7': '963.92'}}, {'source': '', 'values': {'Unnamed: 0': 'S3', 'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001', 'C4': '1324.53', 'C5': '64.98999999999999', 'C6': '134.88', 'C7': '2102.83'}}, {'source': '', 'values': {'Unnamed: 0': 'S4', 'C1': '1254.78', 'C2': '1115.63', 'C3': '52.31', 'C4': '1036.16', 'C5': '892.63', 'C6': '1464.04', 'C7': '1383.41'}}, {'source': '', 'values': {'Unnamed: 0': 'S5', 'C1': '42.9', 'C2': '891.01', 'C3': '1013.94', 'C4': '1128.72', 'C5': '58.91', 'C6': '42.89', 'C7': '1570.31'}}, {'source': '', 'values': {'Unnamed: 0': 'S6', 'C1': '0.7', 'C2': '139.46', 'C3': '70.03', 'C4': '79.15000000000001', 'C5': '1482', 'C6': '0.91', 'C7': '110.46'}}, {'source': '', 'values': {'Unnamed: 0': 'S7', 'C1': '1732.3', 'C2': '1780.44', 'C3': '486.5', 'C4': '523.74', 'C5': '522.08', 'C6': '82.48', 'C7': '826.41'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
warehouses = []
customers = []
fixed_cost = {}
demand = {}
transport_cost = {}
for rec in records:
    vals = rec['values']
    if 'customer' in vals and 'demand' in vals:
        cust = vals['customer']
        customers.append(cust)
        demand[cust] = int(float(vals['demand']))
    elif 'Unnamed: 0' in vals and 'fixed_costs' in vals:
        wh = vals['Unnamed: 0']
        warehouses.append(wh)
        fixed_cost[wh] = float(vals['fixed_costs'])
    elif 'Unnamed: 0' in vals and any((c in vals for c in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'])):
        wh = vals['Unnamed: 0']
        if wh not in transport_cost:
            transport_cost[wh] = {}
        for cust in customers:
            if cust in vals:
                transport_cost[wh][cust] = float(vals[cust])
if set(warehouses) != set(transport_cost.keys()):
    raise ValueError('Mismatch in warehouse identifiers between fixed cost and transport cost tables.')
for wh in warehouses:
    if set(customers) != set(transport_cost[wh].keys()):
        raise ValueError(f'Mismatch in customer identifiers for warehouse {wh} in transport cost table.')
if set(customers) != set(demand.keys()):
    raise ValueError('Mismatch in customer identifiers between demand and transport cost tables.')
m = gp.Model('Bandcamp_Distribution')
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