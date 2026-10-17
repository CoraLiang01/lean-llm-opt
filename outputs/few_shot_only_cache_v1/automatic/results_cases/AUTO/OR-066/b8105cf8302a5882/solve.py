SOURCE_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', 'values': {'customer': 'C1', 'demand': '144'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', 'values': {'customer': 'C2', 'demand': '216'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '105.97'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '85.31'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '2358.39', 'C2': '1492.08'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '0.07000000000000001', 'C2': '52.32'}}]
import gurobipy as gp
from gurobipy import GRB
SOURCE_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', 'values': {'customer': 'C1', 'demand': '144'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', 'values': {'customer': 'C2', 'demand': '216'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '105.97'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '85.31'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '2358.39', 'C2': '1492.08'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '0.07000000000000001', 'C2': '52.32'}}]
suppliers = []
customers = []
demand = {}
fixed_cost = {}
cost = {}
for rec in SOURCE_RECORDS:
    src = rec['source']
    vals = rec['values']
    if src.endswith('demand.csv'):
        cust = vals['customer']
        if cust not in customers:
            customers.append(cust)
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
        for cust in vals:
            if cust != 'Unnamed: 0':
                cost[sup][cust] = float(vals[cust])
                if cust not in customers:
                    customers.append(cust)
for cust in customers:
    if cust not in demand:
        raise ValueError(f'Missing demand for customer {cust}')
for sup in suppliers:
    if sup not in fixed_cost:
        raise ValueError(f'Missing fixed cost for supplier {sup}')
    if sup not in cost:
        raise ValueError(f'Missing transportation costs for supplier {sup}')
    for cust in customers:
        if cust not in cost[sup]:
            raise ValueError(f'Missing transportation cost for supplier {sup}, customer {cust}')
M = sum((demand[cust] for cust in customers))
m = gp.Model('UFLP_Gurobi')
x_vars = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[sup][cust] * x_vars[sup, cust] for sup in suppliers for cust in customers)) + gp.quicksum((fixed_cost[sup] * y_vars[sup] for sup in suppliers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[sup, cust] for sup in suppliers)) == demand[cust] for cust in customers), name='')
m.addConstrs((gp.quicksum((x_vars[sup, cust] for cust in customers)) <= M * y_vars[sup] for sup in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')