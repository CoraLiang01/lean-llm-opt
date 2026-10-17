import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv', dtype=str, keep_default_na=False)
customers = demand_df.iloc[:, 0].str.strip().tolist()
demand = {}
for idx, row in demand_df.iterrows():
    cust = row[0].strip()
    try:
        val = float(row[1])
    except Exception:
        raise ValueError(f'Invalid demand value for {cust}: {row[1]}')
    demand[cust] = val
fixed_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv', dtype=str, keep_default_na=False)
suppliers = fixed_df.iloc[:, 0].str.strip().tolist()
fixed_cost = {}
for idx, row in fixed_df.iterrows():
    sup = row[0].strip()
    try:
        val = float(row[1])
    except Exception:
        raise ValueError(f'Invalid fixed cost for {sup}: {row[1]}')
    fixed_cost[sup] = val
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv', dtype=str, keep_default_na=False)
cost_df.index = cost_df.iloc[:, 0].str.strip()
cost_df = cost_df.drop(cost_df.columns[0], axis=1)
cost = {}
for sup in suppliers:
    if sup not in cost_df.index:
        raise ValueError(f'Supplier {sup} missing in transportation_costs.csv')
    cost[sup] = {}
    for cust in customers:
        if cust not in cost_df.columns:
            raise ValueError(f'Customer {cust} missing in transportation_costs.csv')
        try:
            val = float(cost_df.loc[sup, cust])
        except Exception:
            raise ValueError(f'Invalid cost for {sup},{cust}: {cost_df.loc[sup, cust]}')
        cost[sup][cust] = val
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch in customer keys between demand and customer list')
if set(fixed_cost.keys()) != set(suppliers):
    raise ValueError('Mismatch in supplier keys between fixed_cost and supplier list')
for sup in suppliers:
    if set(cost[sup].keys()) != set(customers):
        raise ValueError(f'Mismatch in cost columns for supplier {sup}')
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