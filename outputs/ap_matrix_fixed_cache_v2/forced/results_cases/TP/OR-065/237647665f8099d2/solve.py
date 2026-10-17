LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv",\n    "values": {\n      "customer": "C1",\n      "demand": "1083"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv",\n    "values": {\n      "customer": "C2",\n      "demand": "776"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv",\n    "values": {\n      "customer": "C3",\n      "demand": "16214"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "fixed_costs": "102.33"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "fixed_costs": "94.92"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv",\n    "values": {\n      "Unnamed: 0": "S3",\n      "fixed_costs": "91.83"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "C1": "1506.22",\n      "C2": "70.90000000000001",\n      "C3": "8.44"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "C1": "1732.65",\n      "C2": "1780.72",\n      "C3": "567.4400000000001"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S3",\n      "C1": "115.66",\n      "C2": "100.76",\n      "C3": "64.68000000000001"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', 'values': {'customer': 'C1', 'demand': '1083'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', 'values': {'customer': 'C2', 'demand': '776'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', 'values': {'customer': 'C3', 'demand': '16214'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
warehouses = []
fixed_cost = {}
for rec in records:
    if rec['source'].endswith('fixed_cost.csv'):
        wh = rec['values']['Unnamed: 0']
        warehouses.append(wh)
        fixed_cost[wh] = float(rec['values']['fixed_costs'])
customers = []
demand = {}
for rec in records:
    if rec['source'].endswith('demand.csv'):
        cust = rec['values']['customer']
        customers.append(cust)
        demand[cust] = float(rec['values']['demand'])
transportation_cost = {wh: {} for wh in warehouses}
for rec in records:
    if rec['source'].endswith('transportation_costs.csv'):
        wh = rec['values']['Unnamed: 0']
        for cust in customers:
            transportation_cost[wh][cust] = float(rec['values'][cust])
if set(fixed_cost.keys()) != set(warehouses):
    raise ValueError('Mismatch in warehouse fixed cost keys')
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch in customer demand keys')
for wh in warehouses:
    if set(transportation_cost[wh].keys()) != set(customers):
        raise ValueError(f'Transportation cost missing for warehouse {wh}')
m = gp.Model('Bandcamp_UFLP')
y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
x_vars = m.addVars(warehouses, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((fixed_cost[wh] * y_vars[wh] for wh in warehouses)) + gp.quicksum((transportation_cost[wh][cust] * x_vars[wh, cust] for wh in warehouses for cust in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[wh, cust] for wh in warehouses)) == demand[cust] for cust in customers), name='')
m.addConstrs((x_vars[wh, cust] <= demand[cust] * y_vars[wh] for wh in warehouses for cust in customers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')