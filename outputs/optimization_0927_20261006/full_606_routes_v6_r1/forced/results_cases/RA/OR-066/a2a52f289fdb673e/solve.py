LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv",\n    "values": {\n      "customer": "C1",\n      "demand": "144"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv",\n    "values": {\n      "customer": "C2",\n      "demand": "216"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "fixed_costs": "105.97"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "fixed_costs": "85.31"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "C1": "2358.39",\n      "C2": "1492.08"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "C1": "0.07000000000000001",\n      "C2": "52.32"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', 'values': {'customer': 'C1', 'demand': '144'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', 'values': {'customer': 'C2', 'demand': '216'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '105.97'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '85.31'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '2358.39', 'C2': '1492.08'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '0.07000000000000001', 'C2': '52.32'}}]
import gurobipy as gp
from gurobipy import GRB
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', 'values': {'customer': 'C1', 'demand': '144'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', 'values': {'customer': 'C2', 'demand': '216'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '105.97'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '85.31'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '2358.39', 'C2': '1492.08'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '0.07000000000000001', 'C2': '52.32'}}]
suppliers = []
fixed_cost = {}
customers = []
demand = {}
transport_cost = {}
for rec in LEGACY_RECORDS:
    src = rec['source']
    vals = rec['values']
    if src.endswith('fixed_cost.csv'):
        s = vals['Unnamed: 0']
        suppliers.append(s)
        fixed_cost[s] = float(vals['fixed_costs'])
    elif src.endswith('demand.csv'):
        c = vals['customer']
        customers.append(c)
        demand[c] = float(vals['demand'])
    elif src.endswith('transportation_costs.csv'):
        s = vals['Unnamed: 0']
        if s not in transport_cost:
            transport_cost[s] = {}
        for c in customers:
            if c in vals:
                transport_cost[s][c] = float(vals[c])
suppliers = list(dict.fromkeys(suppliers))
customers = list(dict.fromkeys(customers))
for s in suppliers:
    if s not in fixed_cost:
        raise ValueError(f'Missing fixed cost for supplier {s}')
    if s not in transport_cost:
        raise ValueError(f'Missing transport cost row for supplier {s}')
    for c in customers:
        if c not in transport_cost[s]:
            raise ValueError(f'Missing transport cost for supplier {s}, customer {c}')
for c in customers:
    if c not in demand:
        raise ValueError(f'Missing demand for customer {c}')
m = gp.Model('facility_location')
y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
x_vars = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((fixed_cost[s] * y_vars[s] for s in suppliers)) + gp.quicksum((transport_cost[s][c] * x_vars[s, c] for s in suppliers for c in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[s, c] for s in suppliers)) == demand[c] for c in customers), name='')
for s in suppliers:
    for c in customers:
        m.addConstr(x_vars[s, c] <= demand[c] * y_vars[s], name=f'link_{s}_{c}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')