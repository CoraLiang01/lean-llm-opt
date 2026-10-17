SOURCE_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv', 'values': {'customer': 'C1', 'demand': '216'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv', 'values': {'customer': 'C2', 'demand': '216'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv', 'values': {'customer': 'C3', 'demand': '216'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv', 'values': {'customer': 'C4', 'demand': '144'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv', 'values': {'customer': 'C5', 'demand': '144'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv', 'values': {'customer': 'C6', 'demand': '144'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '98.88'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '99.73'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv', 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '94.01000000000001'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv', 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '93.77'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv', 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '107.59'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv', 'values': {'Unnamed: 0': 'S6', 'fixed_costs': '112.65'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '0.08', 'C2': '52.33', 'C3': '73.56999999999999', 'C4': '1237.33', 'C5': '0.07000000000000001', 'C6': '112.16'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '46.02', 'C2': '175.23', 'C3': '2026.83', 'C4': '299.89', 'C5': '966.53', 'C6': '1590.42'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'C1': '1031.74', 'C2': '78.13', 'C3': '99.02', 'C4': '277.07', 'C5': '884.45', 'C6': '1800.86'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', 'values': {'Unnamed: 0': 'S4', 'C1': '868.75', 'C2': '94.2', 'C3': '1776.34', 'C4': '285.48', 'C5': '868.85', 'C6': '86.55'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', 'values': {'Unnamed: 0': 'S5', 'C1': '1577', 'C2': '760.15', 'C3': '2090.19', 'C4': '43.2', 'C5': '1577.12', 'C6': '1095.17'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', 'values': {'Unnamed: 0': 'S6', 'C1': '49.14', 'C2': '4.33', 'C3': '2079.57', 'C4': '277.04', 'C5': '1032.01', 'C6': '1543.49'}}]
import gurobipy as gp
from gurobipy import GRB
SOURCE_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv', 'values': {'customer': 'C1', 'demand': '216'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv', 'values': {'customer': 'C2', 'demand': '216'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv', 'values': {'customer': 'C3', 'demand': '216'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv', 'values': {'customer': 'C4', 'demand': '144'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv', 'values': {'customer': 'C5', 'demand': '144'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv', 'values': {'customer': 'C6', 'demand': '144'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '98.88'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '99.73'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv', 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '94.01000000000001'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv', 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '93.77'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv', 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '107.59'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv', 'values': {'Unnamed: 0': 'S6', 'fixed_costs': '112.65'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '0.08', 'C2': '52.33', 'C3': '73.56999999999999', 'C4': '1237.33', 'C5': '0.07000000000000001', 'C6': '112.16'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '46.02', 'C2': '175.23', 'C3': '2026.83', 'C4': '299.89', 'C5': '966.53', 'C6': '1590.42'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'C1': '1031.74', 'C2': '78.13', 'C3': '99.02', 'C4': '277.07', 'C5': '884.45', 'C6': '1800.86'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', 'values': {'Unnamed: 0': 'S4', 'C1': '868.75', 'C2': '94.2', 'C3': '1776.34', 'C4': '285.48', 'C5': '868.85', 'C6': '86.55'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', 'values': {'Unnamed: 0': 'S5', 'C1': '1577', 'C2': '760.15', 'C3': '2090.19', 'C4': '43.2', 'C5': '1577.12', 'C6': '1095.17'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', 'values': {'Unnamed: 0': 'S6', 'C1': '49.14', 'C2': '4.33', 'C3': '2079.57', 'C4': '277.04', 'C5': '1032.01', 'C6': '1543.49'}}]
demand = {}
for rec in SOURCE_RECORDS:
    if rec['source'].endswith('demand.csv'):
        cust = rec['values']['customer']
        val = float(rec['values']['demand'])
        if cust in demand:
            raise ValueError(f'Duplicate demand entry for {cust}')
        demand[cust] = val
fixed_cost = {}
for rec in SOURCE_RECORDS:
    if rec['source'].endswith('fixed_cost.csv'):
        fac = rec['values']['Unnamed: 0']
        val = float(rec['values']['fixed_costs'])
        if fac in fixed_cost:
            raise ValueError(f'Duplicate fixed cost entry for {fac}')
        fixed_cost[fac] = val
cost = {}
for rec in SOURCE_RECORDS:
    if rec['source'].endswith('transportation_costs.csv'):
        fac = rec['values']['Unnamed: 0']
        if fac in cost:
            raise ValueError(f'Duplicate transportation cost entry for {fac}')
        cost[fac] = {}
        for cust in demand:
            if cust not in rec['values']:
                raise ValueError(f'Missing transportation cost for {fac},{cust}')
            cost[fac][cust] = float(rec['values'][cust])
suppliers = sorted(fixed_cost.keys())
customers = sorted(demand.keys())
for i in suppliers:
    if i not in cost:
        raise ValueError(f'Missing transportation cost row for {i}')
    for j in customers:
        if j not in cost[i]:
            raise ValueError(f'Missing transportation cost for {i},{j}')
M = sum(demand.values())
m = gp.Model('Original_RAG_FLP')
x_vars = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in suppliers for j in customers)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in suppliers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in customers)) <= M * y_vars[i] for i in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')