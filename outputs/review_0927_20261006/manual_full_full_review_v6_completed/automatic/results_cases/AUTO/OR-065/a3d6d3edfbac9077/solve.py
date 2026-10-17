LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv",\n    "values": {\n      "customer": "C1",\n      "demand": "1083"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv",\n    "values": {\n      "customer": "C2",\n      "demand": "776"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv",\n    "values": {\n      "customer": "C3",\n      "demand": "16214"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "fixed_costs": "102.33"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "fixed_costs": "94.92"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv",\n    "values": {\n      "Unnamed: 0": "S3",\n      "fixed_costs": "91.83"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "C1": "1506.22",\n      "C2": "70.90000000000001",\n      "C3": "8.44"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "C1": "1732.65",\n      "C2": "1780.72",\n      "C3": "567.4400000000001"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S3",\n      "C1": "115.66",\n      "C2": "100.76",\n      "C3": "64.68000000000001"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', 'values': {'customer': 'C1', 'demand': '1083'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', 'values': {'customer': 'C2', 'demand': '776'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', 'values': {'customer': 'C3', 'demand': '16214'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
warehouses = []
musicians_bands = []
demand = {}
fixed_cost = {}
cost = {}
for rec in records:
    if rec['source'].endswith('demand.csv'):
        cust = rec['values']['customer']
        if cust not in musicians_bands:
            musicians_bands.append(cust)
        demand[cust] = int(rec['values']['demand'])
for rec in records:
    if rec['source'].endswith('fixed_cost.csv'):
        wh = rec['values']['Unnamed: 0']
        if wh not in warehouses:
            warehouses.append(wh)
        fixed_cost[wh] = float(rec['values']['fixed_costs'])
for rec in records:
    if rec['source'].endswith('transportation_costs.csv'):
        wh = rec['values']['Unnamed: 0']
        if wh not in cost:
            cost[wh] = {}
        for mb in musicians_bands:
            if mb in rec['values']:
                cost[wh][mb] = float(rec['values'][mb])
if set(demand.keys()) != set(musicians_bands):
    raise ValueError('Demand keys do not match musicians/bands list')
if set(fixed_cost.keys()) != set(warehouses):
    raise ValueError('Fixed cost keys do not match warehouses list')
for wh in warehouses:
    if wh not in cost or set(cost[wh].keys()) != set(musicians_bands):
        raise ValueError(f'Cost data missing for warehouse {wh}')
M = sum((demand[j] for j in musicians_bands))
m = gp.Model('Bandcamp_FLP')
x_vars = m.addVars(warehouses, musicians_bands, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in warehouses for j in musicians_bands)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in warehouses)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in warehouses)) == demand[j] for j in musicians_bands), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in musicians_bands)) <= M * y_vars[i] for i in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')