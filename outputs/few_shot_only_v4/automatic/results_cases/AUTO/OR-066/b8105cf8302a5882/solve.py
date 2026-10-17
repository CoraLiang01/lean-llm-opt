import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
if demand_df.shape[1] != 2:
    raise ValueError('demand.csv must have exactly two columns: customer and demand')
customers = []
demand = {}
for idx, row in demand_df.iterrows():
    cust = row.iloc[0]
    val = float(row.iloc[1])
    customers.append(cust)
    demand[cust] = val
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
if fixed_cost_df.shape[1] != 2:
    raise ValueError('fixed_cost.csv must have exactly two columns: supplier and fixed_cost')
suppliers = []
fixed_cost = {}
for idx, row in fixed_cost_df.iterrows():
    sup = row.iloc[0]
    val = float(row.iloc[1])
    suppliers.append(sup)
    fixed_cost[sup] = val
trans_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if trans_costs_df.shape[0] != len(suppliers):
    raise ValueError('transportation_costs.csv row count must match number of suppliers')
if trans_costs_df.shape[1] != len(customers) + 1:
    raise ValueError('transportation_costs.csv must have one column per customer plus one for supplier id')
if list(trans_costs_df.columns)[1:] != customers:
    raise ValueError('transportation_costs.csv columns after the first must match customer order in demand.csv')
cost = {}
for idx, row in trans_costs_df.iterrows():
    sup = row.iloc[0]
    if sup not in suppliers:
        raise ValueError(f'Supplier {sup} in transportation_costs.csv not found in fixed_cost.csv')
    cost[sup] = {}
    for j, cust in enumerate(customers):
        val = float(row.iloc[j + 1])
        cost[sup][cust] = val
M = sum((demand[cust] for cust in customers))
m = gp.Model('UFLP')
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