LEGACY_OBSERVATION = 'fixed_cost.csv\nUnnamed: 0,fixed_costs\nS1,98.88\nS2,99.73\nS3,94.01000000000001\nS4,93.77\nS5,107.59\nS6,112.65\n\ntransportation_costs.csv\nUnnamed: 0,C1,C2,C3,C4,C5,C6\nS1,0.08,52.33,73.56999999999999,1237.33,0.07000000000000001,112.16\nS2,46.02,175.23,2026.83,299.89,966.53,1590.42\nS3,1031.74,78.13,99.02,277.07,884.45,1800.86\nS4,868.75,94.2,1776.34,285.48,868.85,86.55\nS5,1577,760.15,2090.19,43.2,1577.12,1095.17\nS6,49.14,4.33,2079.57,277.04,1032.01,1543.49\n\ndemand.csv\ncustomer,demand\nC1,216\nC2,216\nC3,216\nC4,144\nC5,144\nC6,144'
LEGACY_RECORDS = [{'source': 'fixed_cost.csv', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '98.88'}}, {'source': 'fixed_cost.csv', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '99.73'}}, {'source': 'fixed_cost.csv', 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '94.01000000000001'}}, {'source': 'fixed_cost.csv', 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '93.77'}}, {'source': 'fixed_cost.csv', 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '107.59'}}, {'source': 'fixed_cost.csv', 'values': {'Unnamed: 0': 'S6', 'fixed_costs': '112.65'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '0.08', 'C2': '52.33', 'C3': '73.56999999999999', 'C4': '1237.33', 'C5': '0.07000000000000001', 'C6': '112.16'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '46.02', 'C2': '175.23', 'C3': '2026.83', 'C4': '299.89', 'C5': '966.53', 'C6': '1590.42'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'C1': '1031.74', 'C2': '78.13', 'C3': '99.02', 'C4': '277.07', 'C5': '884.45', 'C6': '1800.86'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'S4', 'C1': '868.75', 'C2': '94.2', 'C3': '1776.34', 'C4': '285.48', 'C5': '868.85', 'C6': '86.55'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'S5', 'C1': '1577', 'C2': '760.15', 'C3': '2090.19', 'C4': '43.2', 'C5': '1577.12', 'C6': '1095.17'}}, {'source': 'transportation_costs.csv', 'values': {'Unnamed: 0': 'S6', 'C1': '49.14', 'C2': '4.33', 'C3': '2079.57', 'C4': '277.04', 'C5': '1032.01', 'C6': '1543.49'}}, {'source': 'demand.csv', 'values': {'customer': 'C1', 'demand': '216'}}, {'source': 'demand.csv', 'values': {'customer': 'C2', 'demand': '216'}}, {'source': 'demand.csv', 'values': {'customer': 'C3', 'demand': '216'}}, {'source': 'demand.csv', 'values': {'customer': 'C4', 'demand': '144'}}, {'source': 'demand.csv', 'values': {'customer': 'C5', 'demand': '144'}}, {'source': 'demand.csv', 'values': {'customer': 'C6', 'demand': '144'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
suppliers = []
fixed_costs = {}
for rec in records:
    if rec['source'] == 'fixed_cost.csv':
        sid = rec['values']['Unnamed: 0']
        suppliers.append(sid)
        fixed_costs[sid] = float(rec['values']['fixed_costs'])
stores = []
transportation_costs = {}
for rec in records:
    if rec['source'] == 'transportation_costs.csv':
        sid = rec['values']['Unnamed: 0']
        if not stores:
            stores = [k for k in rec['values'] if k != 'Unnamed: 0']
        transportation_costs[sid] = {}
        for cid in stores:
            transportation_costs[sid][cid] = float(rec['values'][cid])
demand = {}
for rec in records:
    if rec['source'] == 'demand.csv':
        cid = rec['values']['customer']
        demand[cid] = int(rec['values']['demand'])
if set(fixed_costs.keys()) != set(suppliers):
    raise ValueError('Mismatch in fixed_costs and suppliers')
if set(transportation_costs.keys()) != set(suppliers):
    raise ValueError('Mismatch in transportation_costs and suppliers')
for sid in suppliers:
    if set(transportation_costs[sid].keys()) != set(stores):
        raise ValueError(f'Mismatch in transportation_costs for supplier {sid}')
if set(demand.keys()) != set(stores):
    raise ValueError('Mismatch in demand and stores')
M = sum((demand[cid] for cid in stores))
m = gp.Model('Adidas_Supplier_Selection')
y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
x = m.addVars(suppliers, stores, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((fixed_costs[sid] * y[sid] for sid in suppliers)) + gp.quicksum((transportation_costs[sid][cid] * x[sid, cid] for sid in suppliers for cid in stores)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[sid, cid] for sid in suppliers)) == demand[cid] for cid in stores), name='')
m.addConstrs((gp.quicksum((x[sid, cid] for cid in stores)) <= M * y[sid] for sid in suppliers), name='')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')