LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv",\n    "values": {\n      "customer": "C1",\n      "demand": "1083"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv",\n    "values": {\n      "customer": "C2",\n      "demand": "776"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv",\n    "values": {\n      "customer": "C3",\n      "demand": "16214"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "fixed_costs": "102.33"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "fixed_costs": "94.92"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv",\n    "values": {\n      "Unnamed: 0": "S3",\n      "fixed_costs": "91.83"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "C1": "1506.22",\n      "C2": "70.90000000000001",\n      "C3": "8.44"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "C1": "1732.65",\n      "C2": "1780.72",\n      "C3": "567.4400000000001"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S3",\n      "C1": "115.66",\n      "C2": "100.76",\n      "C3": "64.68000000000001"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', 'values': {'customer': 'C1', 'demand': '1083'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', 'values': {'customer': 'C2', 'demand': '776'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', 'values': {'customer': 'C3', 'demand': '16214'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001'}}]
import gurobipy as gp
from gurobipy import GRB
demand = {}
fixed_costs = {}
transportation_costs = {}
warehouses = []
customers = []
for rec in LEGACY_RECORDS:
    src = rec['source']
    vals = rec['values']
    if src.endswith('demand.csv'):
        cust = vals['customer']
        customers.append(cust)
        demand[cust] = float(vals['demand'])
    elif src.endswith('fixed_cost.csv'):
        wh = vals['Unnamed: 0']
        warehouses.append(wh)
        fixed_costs[wh] = float(vals['fixed_costs'])
    elif src.endswith('transportation_costs.csv'):
        wh = vals['Unnamed: 0']
        if wh not in transportation_costs:
            transportation_costs[wh] = {}
        for cust in ['C1', 'C2', 'C3']:
            transportation_costs[wh][cust] = float(vals[cust])
warehouses = list(dict.fromkeys(warehouses))
customers = list(dict.fromkeys(customers))
for wh in warehouses:
    if wh not in fixed_costs:
        raise ValueError(f'Missing fixed cost for warehouse {wh}')
    if wh not in transportation_costs:
        raise ValueError(f'Missing transportation costs for warehouse {wh}')
    for cust in customers:
        if cust not in transportation_costs[wh]:
            raise ValueError(f'Missing transportation cost for warehouse {wh}, customer {cust}')
for cust in customers:
    if cust not in demand:
        raise ValueError(f'Missing demand for customer {cust}')
M = sum((demand[cust] for cust in customers))
M_i = {wh: M for wh in warehouses}
m = gp.Model('Bandcamp_UFLP')
y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
x_vars = m.addVars(warehouses, customers, vtype=GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((fixed_costs[wh] * y_vars[wh] for wh in warehouses)) + gp.quicksum((transportation_costs[wh][cust] * x_vars[wh, cust] for wh in warehouses for cust in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[wh, cust] for wh in warehouses)) == demand[cust] for cust in customers), name='')
m.addConstrs((gp.quicksum((x_vars[wh, cust] for cust in customers)) <= M_i[wh] * y_vars[wh] for wh in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')