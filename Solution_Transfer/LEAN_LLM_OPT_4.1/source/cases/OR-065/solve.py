LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv",\n    "values": {\n      "customer": "C1",\n      "demand": "1083"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv",\n    "values": {\n      "customer": "C2",\n      "demand": "776"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv",\n    "values": {\n      "customer": "C3",\n      "demand": "16214"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "fixed_costs": "102.33"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "fixed_costs": "94.92"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv",\n    "values": {\n      "Unnamed: 0": "S3",\n      "fixed_costs": "91.83"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "C1": "1506.22",\n      "C2": "70.90000000000001",\n      "C3": "8.44"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "C1": "1732.65",\n      "C2": "1780.72",\n      "C3": "567.4400000000001"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S3",\n      "C1": "115.66",\n      "C2": "100.76",\n      "C3": "64.68000000000001"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', 'values': {'customer': 'C1', 'demand': '1083'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', 'values': {'customer': 'C2', 'demand': '776'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', 'values': {'customer': 'C3', 'demand': '16214'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
warehouses = []
musicians = []
demand = {}
fixed_cost = {}
cost = {}
for rec in records:
    src = rec['source']
    vals = rec['values']
    if src.endswith('demand.csv'):
        cust = vals['customer']
        if cust not in musicians:
            musicians.append(cust)
        demand[cust] = float(vals['demand'])
    elif src.endswith('fixed_cost.csv'):
        wh = vals['Unnamed: 0']
        if wh not in warehouses:
            warehouses.append(wh)
        fixed_cost[wh] = float(vals['fixed_costs'])
    elif src.endswith('transportation_costs.csv'):
        wh = vals['Unnamed: 0']
        if wh not in warehouses:
            warehouses.append(wh)
        if wh not in cost:
            cost[wh] = {}
        for cust in musicians or [k for k in vals if k != 'Unnamed: 0']:
            if cust in vals:
                cost[wh][cust] = float(vals[cust])
for wh in warehouses:
    if wh not in cost:
        raise ValueError(f'Missing transportation cost row for warehouse {wh}')
    for cust in musicians:
        if cust not in cost[wh]:
            raise ValueError(f'Missing transportation cost for warehouse {wh}, customer {cust}')
M = sum((demand[cust] for cust in musicians))
m = gp.Model('Bandcamp_FLP')
x_vars = m.addVars(warehouses, musicians, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[wh][cust] * x_vars[wh, cust] for wh in warehouses for cust in musicians)) + gp.quicksum((fixed_cost[wh] * y_vars[wh] for wh in warehouses)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[wh, cust] for wh in warehouses)) == demand[cust] for cust in musicians), name='')
m.addConstrs((gp.quicksum((x_vars[wh, cust] for cust in musicians)) <= M * y_vars[wh] for wh in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')