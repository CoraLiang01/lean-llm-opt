LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv","values":{"customer":"C1","demand":"1083"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv","values":{"customer":"C2","demand":"776"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv","values":{"customer":"C3","demand":"16214"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv","values":{"Unnamed: 0":"S1","fixed_costs":"102.33"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv","values":{"Unnamed: 0":"S2","fixed_costs":"94.92"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv","values":{"Unnamed: 0":"S3","fixed_costs":"91.83"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv","values":{"Unnamed: 0":"S1","C1":"1506.22","C2":"70.90000000000001","C3":"8.44"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv","values":{"Unnamed: 0":"S2","C1":"1732.65","C2":"1780.72","C3":"567.4400000000001"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv","values":{"Unnamed: 0":"S3","C1":"115.66","C2":"100.76","C3":"64.68000000000001"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', 'values': {'customer': 'C1', 'demand': '1083'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', 'values': {'customer': 'C2', 'demand': '776'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', 'values': {'customer': 'C3', 'demand': '16214'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001'}}]
import gurobipy as gp
from gurobipy import GRB
warehouses = []
customers = []
fixed_cost = {}
demand = {}
transport_cost = {}
for rec in LEGACY_RECORDS:
    src = rec['source']
    vals = rec['values']
    if src.endswith('fixed_cost.csv'):
        wh = vals['Unnamed: 0']
        warehouses.append(wh)
        fixed_cost[wh] = float(vals['fixed_costs'])
    elif src.endswith('demand.csv'):
        cust = vals['customer']
        customers.append(cust)
        demand[cust] = int(vals['demand'])
    elif src.endswith('transportation_costs.csv'):
        wh = vals['Unnamed: 0']
        if wh not in transport_cost:
            transport_cost[wh] = {}
        for cust in vals:
            if cust != 'Unnamed: 0':
                transport_cost[wh][cust] = float(vals[cust])
warehouses = list(dict.fromkeys(warehouses))
customers = list(dict.fromkeys(customers))
for wh in warehouses:
    if wh not in fixed_cost:
        raise ValueError(f'Missing fixed cost for warehouse {wh}')
    if wh not in transport_cost:
        raise ValueError(f'Missing transport cost row for warehouse {wh}')
    for cust in customers:
        if cust not in transport_cost[wh]:
            raise ValueError(f'Missing transport cost for warehouse {wh}, customer {cust}')
for cust in customers:
    if cust not in demand:
        raise ValueError(f'Missing demand for customer {cust}')
M = {cust: demand[cust] for cust in customers}
m = gp.Model('Bandcamp_Warehouse_Selection')
y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
x_vars = m.addVars(warehouses, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((fixed_cost[wh] * y_vars[wh] for wh in warehouses)) + gp.quicksum((transport_cost[wh][cust] * x_vars[wh, cust] for wh in warehouses for cust in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[wh, cust] for wh in warehouses)) == demand[cust] for cust in customers), name='')
m.addConstrs((x_vars[wh, cust] <= M[cust] * y_vars[wh] for wh in warehouses for cust in customers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')