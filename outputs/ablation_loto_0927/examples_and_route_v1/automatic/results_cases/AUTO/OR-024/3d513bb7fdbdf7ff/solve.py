LEGACY_OBSERVATION = '{"values": {"Product Name": "S700_1138", "Revenue": "70.67", "Demand": "1219", "Initial Inventory": "9020"}}\n{"values": {"Product Name": "S700_1691", "Revenue": "100.0", "Demand": "1127", "Initial Inventory": "8370"}}\n{"values": {"Product Name": "S700_1938", "Revenue": "70.15", "Demand": "1129", "Initial Inventory": "8390"}}\n{"values": {"Product Name": "S700_2047", "Revenue": "100.0", "Demand": "1176", "Initial Inventory": "8680"}}\n{"values": {"Product Name": "S700_2466", "Revenue": "100.0", "Demand": "1301", "Initial Inventory": "9400"}}\n{"values": {"Product Name": "S700_2610", "Revenue": "65.77", "Demand": "1340", "Initial Inventory": "9900"}}\n{"values": {"Product Name": "S700_2824", "Revenue": "100.0", "Demand": "1357", "Initial Inventory": "9760"}}\n{"values": {"Product Name": "S700_2834", "Revenue": "100.0", "Demand": "1158", "Initial Inventory": "8610"}}\n{"values": {"Product Name": "S700_3167", "Revenue": "74.4", "Demand": "1287", "Initial Inventory": "9380"}}\n{"values": {"Product Name": "S700_3505", "Revenue": "81.14", "Demand": "1281", "Initial Inventory": "9170"}}\n{"values": {"Product Name": "S700_3962", "Revenue": "100.0", "Demand": "1135", "Initial Inventory": "8520"}}\n{"values": {"Product Name": "S700_4002", "Revenue": "61.44", "Demand": "1392", "Initial Inventory": "10290"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': 'S700_1138', 'Revenue': '70.67', 'Demand': '1219', 'Initial Inventory': '9020'}}, {'source': '', 'values': {'Product Name': 'S700_1691', 'Revenue': '100.0', 'Demand': '1127', 'Initial Inventory': '8370'}}, {'source': '', 'values': {'Product Name': 'S700_1938', 'Revenue': '70.15', 'Demand': '1129', 'Initial Inventory': '8390'}}, {'source': '', 'values': {'Product Name': 'S700_2047', 'Revenue': '100.0', 'Demand': '1176', 'Initial Inventory': '8680'}}, {'source': '', 'values': {'Product Name': 'S700_2466', 'Revenue': '100.0', 'Demand': '1301', 'Initial Inventory': '9400'}}, {'source': '', 'values': {'Product Name': 'S700_2610', 'Revenue': '65.77', 'Demand': '1340', 'Initial Inventory': '9900'}}, {'source': '', 'values': {'Product Name': 'S700_2824', 'Revenue': '100.0', 'Demand': '1357', 'Initial Inventory': '9760'}}, {'source': '', 'values': {'Product Name': 'S700_2834', 'Revenue': '100.0', 'Demand': '1158', 'Initial Inventory': '8610'}}, {'source': '', 'values': {'Product Name': 'S700_3167', 'Revenue': '74.4', 'Demand': '1287', 'Initial Inventory': '9380'}}, {'source': '', 'values': {'Product Name': 'S700_3505', 'Revenue': '81.14', 'Demand': '1281', 'Initial Inventory': '9170'}}, {'source': '', 'values': {'Product Name': 'S700_3962', 'Revenue': '100.0', 'Demand': '1135', 'Initial Inventory': '8520'}}, {'source': '', 'values': {'Product Name': 'S700_4002', 'Revenue': '61.44', 'Demand': '1392', 'Initial Inventory': '10290'}}]
import gurobipy as gp
from gurobipy import GRB
I = []
r = {}
d = {}
s = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    pid = v['Product Name']
    if pid.startswith('S700_'):
        I.append(pid)
        try:
            r[pid] = float(v['Revenue'])
            d[pid] = int(v['Demand'])
            s[pid] = int(v['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Missing or invalid data for product {pid}: {e}')
for pid in I:
    if pid not in r or pid not in d or pid not in s:
        raise ValueError(f'Missing data for product {pid}')
ub = {pid: min(d[pid], s[pid]) for pid in I}
m = gp.Model('S700_Fulfillment')
x = m.addVars(I, lb=0, ub=ub, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((r[pid] * x[pid] for pid in I)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')