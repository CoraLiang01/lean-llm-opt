LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv",\n    "values": {\n      "customer": "C1",\n      "demand": "144"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv",\n    "values": {\n      "customer": "C2",\n      "demand": "216"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "fixed_costs": "105.97"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "fixed_costs": "85.31"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "C1": "2358.39",\n      "C2": "1492.08"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "C1": "0.07000000000000001",\n      "C2": "52.32"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', 'values': {'customer': 'C1', 'demand': '144'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', 'values': {'customer': 'C2', 'demand': '216'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '105.97'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '85.31'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '2358.39', 'C2': '1492.08'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '0.07000000000000001', 'C2': '52.32'}}]
import gurobipy as gp
from gurobipy import GRB
suppliers = []
supermarkets = []
demand = {}
fixed_cost = {}
cost = {}
for rec in LEGACY_RECORDS:
    src = rec['source']
    vals = rec['values']
    if src.endswith('demand.csv'):
        cust = vals['customer']
        if cust not in supermarkets:
            supermarkets.append(cust)
        demand[cust] = float(vals['demand'])
    elif src.endswith('fixed_cost.csv'):
        sup = vals['Unnamed: 0']
        if sup not in suppliers:
            suppliers.append(sup)
        fixed_cost[sup] = float(vals['fixed_costs'])
    elif src.endswith('transportation_costs.csv'):
        sup = vals['Unnamed: 0']
        if sup not in suppliers:
            suppliers.append(sup)
        if sup not in cost:
            cost[sup] = {}
        for j in vals:
            if j != 'Unnamed: 0':
                cost[sup][j] = float(vals[j])
for j in supermarkets:
    if j not in demand:
        raise ValueError(f'Missing demand for supermarket {j}')
for i in suppliers:
    if i not in fixed_cost:
        raise ValueError(f'Missing fixed cost for supplier {i}')
    if i not in cost:
        raise ValueError(f'Missing cost row for supplier {i}')
    for j in supermarkets:
        if j not in cost[i]:
            raise ValueError(f'Missing transportation cost for supplier {i}, supermarket {j}')
M = sum((demand[j] for j in supermarkets))
m = gp.Model('UFLP_Gurobi')
x_vars = m.addVars(suppliers, supermarkets, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in suppliers for j in supermarkets)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in suppliers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in suppliers)) == demand[j] for j in supermarkets), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in supermarkets)) <= M * y_vars[i] for i in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')