import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv', dtype=str, keep_default_na=False)
customers = demand_df['customer'].tolist()
demand = {}
for idx, row in demand_df.iterrows():
    cust = row['customer']
    try:
        val = float(row['demand'])
    except Exception:
        raise ValueError(f'Invalid demand value for customer {cust}')
    demand[cust] = val
fixed_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv', dtype=str, keep_default_na=False)
suppliers = fixed_df['Unnamed: 0'].tolist()
fixed_cost = {}
for idx, row in fixed_df.iterrows():
    sup = row['Unnamed: 0']
    try:
        val = float(row['fixed_costs'])
    except Exception:
        raise ValueError(f'Invalid fixed cost for supplier {sup}')
    fixed_cost[sup] = val
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv', dtype=str, keep_default_na=False)
cost = {}
for idx, row in costs_df.iterrows():
    sup = row['Unnamed: 0']
    cost[sup] = {}
    for cust in customers:
        try:
            val = float(row[cust])
        except Exception:
            raise ValueError(f'Invalid transportation cost for supplier {sup}, customer {cust}')
        cost[sup][cust] = val
if set(cost.keys()) != set(suppliers):
    raise ValueError('Mismatch between suppliers in fixed_cost and transportation_costs')
for sup in suppliers:
    if set(cost[sup].keys()) != set(customers):
        raise ValueError(f'Mismatch in customers for supplier {sup} in transportation_costs')
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch between customers in demand and transportation_costs')
M = sum((demand[cust] for cust in customers))
m = gp.Model('UFLP5')
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