LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv",\n    "values": {\n      "customer": "C1",\n      "demand": "144"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv",\n    "values": {\n      "customer": "C2",\n      "demand": "216"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "fixed_costs": "105.97"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "fixed_costs": "85.31"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "C1": "2358.39",\n      "C2": "1492.08"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "C1": "0.07000000000000001",\n      "C2": "52.32"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', 'values': {'customer': 'C1', 'demand': '144'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', 'values': {'customer': 'C2', 'demand': '216'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '105.97'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '85.31'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '2358.39', 'C2': '1492.08'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '0.07000000000000001', 'C2': '52.32'}}]
import gurobipy as gp
from gurobipy import GRB
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', 'values': {'customer': 'C1', 'demand': '144'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', 'values': {'customer': 'C2', 'demand': '216'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '105.97'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '85.31'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '2358.39', 'C2': '1492.08'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '0.07000000000000001', 'C2': '52.32'}}]
suppliers = []
fixed_costs = {}
customers = []
demands = {}
transportation_costs = {}
for rec in LEGACY_RECORDS:
    src = rec['source']
    vals = rec['values']
    if src.endswith('fixed_cost.csv'):
        s = vals['Unnamed: 0']
        suppliers.append(s)
        fixed_costs[s] = float(vals['fixed_costs'])
    elif src.endswith('demand.csv'):
        c = vals['customer']
        customers.append(c)
        demands[c] = float(vals['demand'])
    elif src.endswith('transportation_costs.csv'):
        s = vals['Unnamed: 0']
        if s not in transportation_costs:
            transportation_costs[s] = {}
        for c in vals:
            if c != 'Unnamed: 0':
                transportation_costs[s][c] = float(vals[c])
suppliers = list(dict.fromkeys(suppliers))
customers = list(dict.fromkeys(customers))
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
total_demand = sum((demands[c] for c in customers))
m = gp.Model('UFLP')
m.Params.MIPGap = 0.0001
y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
x_vars = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((fixed_costs[s] * y_vars[s] for s in suppliers)) + gp.quicksum((transportation_costs[s][c] * x_vars[s, c] for s in suppliers for c in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[s, c] for s in suppliers)) == demands[c] for c in customers), name='')
m.addConstrs((gp.quicksum((x_vars[s, c] for c in customers)) <= total_demand * y_vars[s] for s in suppliers), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')