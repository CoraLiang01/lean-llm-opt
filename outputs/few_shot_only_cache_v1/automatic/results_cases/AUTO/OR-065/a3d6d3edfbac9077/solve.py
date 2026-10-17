SOURCE_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', 'values': {'customer': 'C1', 'demand': '1083'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', 'values': {'customer': 'C2', 'demand': '776'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', 'values': {'customer': 'C3', 'demand': '16214'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001'}}]
import gurobipy as gp
from gurobipy import GRB
demand = {}
fixed_cost = {}
cost = {}
facilities = []
customers = []
for rec in SOURCE_RECORDS:
    src = rec['source']
    vals = rec['values']
    if src.endswith('demand.csv'):
        cust = vals['customer']
        dem = float(vals['demand'])
        demand[cust] = dem
        if cust not in customers:
            customers.append(cust)
    elif src.endswith('fixed_cost.csv'):
        fac = vals['Unnamed: 0']
        fc = float(vals['fixed_costs'])
        fixed_cost[fac] = fc
        if fac not in facilities:
            facilities.append(fac)
    elif src.endswith('transportation_costs.csv'):
        fac = vals['Unnamed: 0']
        if fac not in cost:
            cost[fac] = {}
        for cust in vals:
            if cust == 'Unnamed: 0':
                continue
            cost[fac][cust] = float(vals[cust])
        if fac not in facilities:
            facilities.append(fac)
        for cust in vals:
            if cust != 'Unnamed: 0' and cust not in customers:
                customers.append(cust)
for fac in facilities:
    if fac not in fixed_cost:
        raise ValueError(f'Missing fixed cost for facility {fac}')
    if fac not in cost:
        raise ValueError(f'Missing transportation cost row for facility {fac}')
    for cust in customers:
        if cust not in cost[fac]:
            raise ValueError(f'Missing transportation cost for facility {fac}, customer {cust}')
for cust in customers:
    if cust not in demand:
        raise ValueError(f'Missing demand for customer {cust}')
M = sum((demand[cust] for cust in customers))
m = gp.Model('Bandcamp_Warehouse_Selection')
x_vars = m.addVars(facilities, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(facilities, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[fac][cust] * x_vars[fac, cust] for fac in facilities for cust in customers)) + gp.quicksum((fixed_cost[fac] * y_vars[fac] for fac in facilities)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[fac, cust] for fac in facilities)) == demand[cust] for cust in customers), name='')
m.addConstrs((gp.quicksum((x_vars[fac, cust] for cust in customers)) <= M * y_vars[fac] for fac in facilities), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')