import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv', dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv', dtype=str, keep_default_na=False)
transport_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv', dtype=str, keep_default_na=False)
suppliers = list(fixed_cost_df['supplier'])
customers = list(demand_df['customer'])
demand = {}
for _, row in demand_df.iterrows():
    cust = row['customer']
    try:
        demand[cust] = float(row['demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
fixed_cost = {}
for _, row in fixed_cost_df.iterrows():
    sup = row['supplier']
    try:
        fixed_cost[sup] = float(row['fixed_cost'])
    except Exception:
        raise ValueError(f"Invalid fixed cost for supplier {sup}: {row['fixed_cost']}")
cost = {}
for _, row in transport_cost_df.iterrows():
    sup = row['supplier']
    cost[sup] = {}
    for cust in customers:
        try:
            cost_val = float(row[cust])
        except Exception:
            raise ValueError(f'Invalid transportation cost for supplier {sup}, customer {cust}: {row[cust]}')
        cost[sup][cust] = cost_val
if set(cost.keys()) != set(suppliers):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
for sup in suppliers:
    if set(cost[sup].keys()) != set(customers):
        raise ValueError(f'Mismatch between customers in transportation_costs.csv for supplier {sup}')
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch between customers in demand.csv and transportation_costs.csv')
M = sum((demand[cust] for cust in customers))
m = gp.Model('UFLP3')
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