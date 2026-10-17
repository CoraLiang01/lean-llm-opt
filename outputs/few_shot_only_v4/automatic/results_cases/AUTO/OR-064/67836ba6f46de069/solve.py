import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/demand.csv', dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/fixed_cost.csv', dtype=str, keep_default_na=False)
transport_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/transportation_costs.csv', dtype=str, keep_default_na=False)
customers = list(demand_df['Customer'])
suppliers = list(fixed_cost_df['Supplier'])
try:
    demand = {row['Customer']: float(row['Demand']) for _, row in demand_df.iterrows()}
except Exception as e:
    raise ValueError(f'Error parsing demand.csv: {e}')
try:
    fixed_cost = {row['Supplier']: float(row['Fixed Cost']) for _, row in fixed_cost_df.iterrows()}
except Exception as e:
    raise ValueError(f'Error parsing fixed_cost.csv: {e}')
try:
    cost = {}
    for _, row in transport_cost_df.iterrows():
        supplier = row['']
        cost[supplier] = {}
        for customer in customers:
            val = row[customer]
            if val == '':
                raise ValueError(f'Missing transportation cost for supplier {supplier}, customer {customer}')
            cost[supplier][customer] = float(val)
except Exception as e:
    raise ValueError(f'Error parsing transportation_costs.csv: {e}')
if set(suppliers) != set(cost.keys()):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
for s in suppliers:
    if set(customers) != set(cost[s].keys()):
        raise ValueError(f'Mismatch between customers in demand.csv and transportation_costs.csv for supplier {s}')
M = sum((demand[c] for c in customers))
m = gp.Model('UFLP8')
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