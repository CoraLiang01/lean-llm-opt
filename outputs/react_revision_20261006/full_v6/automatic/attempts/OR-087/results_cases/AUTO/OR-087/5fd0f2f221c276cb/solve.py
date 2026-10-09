import gurobipy as gp
import pandas as pd
import numpy as np
df_36_1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv', dtype=str, keep_default_na=False)
df_36_2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv', dtype=str, keep_default_na=False)
df_36_3 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv', dtype=str, keep_default_na=False)
product_ids = [col for col in df_36_1.columns if col != 'Product']
if len(product_ids) != 80:
    raise ValueError(f'Expected 80 products, found {len(product_ids)}')

def get_row(df, row_name):

    def norm(s):
        return ' '.join(s.strip().split()).casefold()
    norm_row_name = norm(row_name)
    for (idx, val) in df['Product'].items():
        if norm(val) == norm_row_name:
            return df.loc[idx]
    raise KeyError(f"Row '{row_name}' not found in DataFrame.")
row_max_demand = get_row(df_36_1, 'Maximum Demand (100 kg units)')
row_selling_price = get_row(df_36_1, 'Selling Price ($/100 kg)')
row_prod_cost = get_row(df_36_1, 'Production Cost ($/100 kg)')
row_daily_quota = get_row(df_36_1, 'Production Quota (max per day)')
row_activation_cost = get_row(df_36_2, 'Activation Cost ($)')
row_min_batch = get_row(df_36_3, 'Minimum Batch Size (100 kg units)')

def to_float_dict(row):
    return {pid: float(row[pid]) for pid in product_ids}

def to_int_dict(row):
    return {pid: int(float(row[pid])) for pid in product_ids}
max_demand = to_float_dict(row_max_demand)
selling_price = to_float_dict(row_selling_price)
prod_cost = to_float_dict(row_prod_cost)
daily_quota = to_float_dict(row_daily_quota)
activation_cost = to_float_dict(row_activation_cost)
min_batch = to_int_dict(row_min_batch)

def solve_problem():
    m = gp.Model('MonthlyProductionPlan')
    x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    y_vars = m.addVars(product_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum(((selling_price[i] - prod_cost[i]) * x_vars[i] - activation_cost[i] * y_vars[i] for i in product_ids)), gp.GRB.MAXIMIZE)
    m.addConstrs((x_vars[i] <= max_demand[i] for i in product_ids), name='')
    m.addConstrs((x_vars[i] <= 22 * daily_quota[i] for i in product_ids), name='')
    m.addConstrs((x_vars[i] >= min_batch[i] * y_vars[i] for i in product_ids), name='')
    m.addConstrs((x_vars[i] <= max_demand[i] * y_vars[i] for i in product_ids), name='')
    m.addConstr(gp.quicksum((x_vars[i] / daily_quota[i] for i in product_ids)) <= 22, name='shared_days')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.4f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')