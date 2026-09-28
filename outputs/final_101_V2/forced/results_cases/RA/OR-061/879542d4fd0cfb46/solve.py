LEGACY_OBSERVATION = '{"values": {"customer": "C1", "demand": "143"}}\n{"values": {"customer": "C2", "demand": "6"}}\n{"values": {"customer": "C3", "demand": "10"}}\n{"values": {"customer": "C4", "demand": "25"}}\n{"values": {"customer": "C5", "demand": "3"}}\n{"values": {"Unnamed: 0": "S1", "fixed_costs": "97.65000000000001"}}\n{"values": {"Unnamed: 0": "S2", "fixed_costs": "99.76000000000001"}}\n{"values": {"Unnamed: 0": "S3", "fixed_costs": "100.76"}}\n{"values": {"Unnamed: 0": "S4", "fixed_costs": "105.32"}}\n{"values": {"Unnamed: 0": "S5", "fixed_costs": "98.88"}}\n{"values": {"Unnamed: 0": "S1", "C1": "150.74", "C2": "0.02", "C3": "49.13", "C4": "2080.15", "C5": "426.4"}}\n{"values": {"Unnamed: 0": "S2", "C1": "233.05", "C2": "97.73", "C3": "49.84", "C4": "1982.39", "C5": "23.96"}}\n{"values": {"Unnamed: 0": "S3", "C1": "55.68", "C2": "935.61", "C3": "4.03", "C4": "73.09", "C5": "525.3200000000001"}}\n{"values": {"Unnamed: 0": "S4", "C1": "1483.82", "C2": "1801.08", "C3": "112.16", "C4": "816.05", "C5": "107.01"}}\n{"values": {"Unnamed: 0": "S5", "C1": "1119.47", "C2": "884.3099999999999", "C3": "0.08", "C4": "1544.95", "C5": "543.67"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'customer': 'C1', 'demand': '143'}}, {'source': '', 'values': {'customer': 'C2', 'demand': '6'}}, {'source': '', 'values': {'customer': 'C3', 'demand': '10'}}, {'source': '', 'values': {'customer': 'C4', 'demand': '25'}}, {'source': '', 'values': {'customer': 'C5', 'demand': '3'}}, {'source': '', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '97.65000000000001'}}, {'source': '', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '99.76000000000001'}}, {'source': '', 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '100.76'}}, {'source': '', 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '105.32'}}, {'source': '', 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '98.88'}}, {'source': '', 'values': {'Unnamed: 0': 'S1', 'C1': '150.74', 'C2': '0.02', 'C3': '49.13', 'C4': '2080.15', 'C5': '426.4'}}, {'source': '', 'values': {'Unnamed: 0': 'S2', 'C1': '233.05', 'C2': '97.73', 'C3': '49.84', 'C4': '1982.39', 'C5': '23.96'}}, {'source': '', 'values': {'Unnamed: 0': 'S3', 'C1': '55.68', 'C2': '935.61', 'C3': '4.03', 'C4': '73.09', 'C5': '525.3200000000001'}}, {'source': '', 'values': {'Unnamed: 0': 'S4', 'C1': '1483.82', 'C2': '1801.08', 'C3': '112.16', 'C4': '816.05', 'C5': '107.01'}}, {'source': '', 'values': {'Unnamed: 0': 'S5', 'C1': '1119.47', 'C2': '884.3099999999999', 'C3': '0.08', 'C4': '1544.95', 'C5': '543.67'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
demand = {}
for rec in records:
    v = rec['values']
    if 'customer' in v and 'demand' in v:
        demand[v['customer']] = float(v['demand'])
fixed_cost = {}
for rec in records:
    v = rec['values']
    if 'Unnamed: 0' in v and 'fixed_costs' in v:
        fixed_cost[v['Unnamed: 0']] = float(v['fixed_costs'])
transport_cost = {}
for rec in records:
    v = rec['values']
    if 'Unnamed: 0' in v and any((k in v for k in demand)):
        i = v['Unnamed: 0']
        transport_cost[i] = {}
        for j in demand:
            if j in v:
                transport_cost[i][j] = float(v[j])
I = sorted(fixed_cost.keys())
J = sorted(demand.keys())
for i in I:
    if i not in transport_cost or any((j not in transport_cost[i] for j in J)):
        raise ValueError(f'Missing transportation cost for supplier {i}')
for j in J:
    if j not in demand:
        raise ValueError(f'Missing demand for customer {j}')
for i in I:
    if i not in fixed_cost:
        raise ValueError(f'Missing fixed cost for supplier {i}')
m = gp.Model('Superstore_Supplier_Location')
y = m.addVars(I, vtype=GRB.BINARY, name='')
x = m.addVars(I, J, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in I)) + gp.quicksum((transport_cost[i][j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in I)) == demand[j] for j in J), name='')
m.addConstrs((x[i, j] <= demand[j] * y[i] for i in I for j in J), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')