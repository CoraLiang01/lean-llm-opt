LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv",\n    "values": {\n      "customer": "C1",\n      "demand": "144"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv",\n    "values": {\n      "customer": "C2",\n      "demand": "216"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "fixed_costs": "105.97"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "fixed_costs": "85.31"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "C1": "2358.39",\n      "C2": "1492.08"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "C1": "0.07000000000000001",\n      "C2": "52.32"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', 'values': {'customer': 'C1', 'demand': '144'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', 'values': {'customer': 'C2', 'demand': '216'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '105.97'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '85.31'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '2358.39', 'C2': '1492.08'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '0.07000000000000001', 'C2': '52.32'}}]
import gurobipy as gp
from gurobipy import GRB
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', 'values': {'customer': 'C1', 'demand': '144'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', 'values': {'customer': 'C2', 'demand': '216'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '105.97'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '85.31'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '2358.39', 'C2': '1492.08'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '0.07000000000000001', 'C2': '52.32'}}]
suppliers = []
fixed_costs = {}
for rec in LEGACY_RECORDS:
    if rec['source'].endswith('fixed_cost.csv'):
        s = rec['values']['Unnamed: 0']
        suppliers.append(s)
        fixed_costs[s] = float(rec['values']['fixed_costs'])
customers = []
demands = {}
for rec in LEGACY_RECORDS:
    if rec['source'].endswith('demand.csv'):
        c = rec['values']['customer']
        customers.append(c)
        demands[c] = float(rec['values']['demand'])
transportation_costs = {s: {} for s in suppliers}
for rec in LEGACY_RECORDS:
    if rec['source'].endswith('transportation_costs.csv'):
        s = rec['values']['Unnamed: 0']
        for c in customers:
            transportation_costs[s][c] = float(rec['values'][c])
for s in suppliers:
    if s not in fixed_costs:
        raise ValueError(f'Missing fixed cost for supplier {s}')
    if s not in transportation_costs:
        raise ValueError(f'Missing transportation costs for supplier {s}')
    for c in customers:
        if c not in transportation_costs[s]:
            raise ValueError(f'Missing transportation cost for supplier {s} to customer {c}')
for c in customers:
    if c not in demands:
        raise ValueError(f'Missing demand for customer {c}')
m = gp.Model('UFLP_Gurobi')
y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
x_vars = m.addVars(suppliers, customers, vtype=GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((fixed_costs[s] * y_vars[s] for s in suppliers)) + gp.quicksum((transportation_costs[s][c] * x_vars[s, c] for s in suppliers for c in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[s, c] for s in suppliers)) == demands[c] for c in customers), name='')
m.addConstrs((x_vars[s, c] <= demands[c] * y_vars[s] for s in suppliers for c in customers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')