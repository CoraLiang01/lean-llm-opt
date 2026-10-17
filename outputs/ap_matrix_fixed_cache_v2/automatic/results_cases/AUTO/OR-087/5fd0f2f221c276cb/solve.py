import gurobipy as gp
import pandas as pd
import numpy as np
df_36_1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv', dtype=str, keep_default_na=False)
df_36_2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv', dtype=str, keep_default_na=False)
df_36_3 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv', dtype=str, keep_default_na=False)
product_ids = [col for col in df_36_1.columns if col != 'Product']
if len(product_ids) != 80:
    raise ValueError(f'Expected 80 products, found {len(product_ids)}: {product_ids}')

def get_row_by_label(df, label):
    norm_label = label.strip().casefold()
    for idx, val in enumerate(df['Product']):
        if val.strip().casefold() == norm_label:
            return df.iloc[idx]
    raise KeyError(f"Row '{label}' not found in DataFrame.")
row_max_demand = get_row_by_label(df_36_1, 'Maximum Demand (100 kg units)')
row_selling_price = get_row_by_label(df_36_1, 'Selling Price ($/100 kg)')
row_prod_cost = get_row_by_label(df_36_1, 'Production Cost ($/100 kg)')
row_daily_quota = get_row_by_label(df_36_1, 'Production Quota (max per day)')
row_activation_cost = get_row_by_label(df_36_2, 'Activation Cost ($)')
row_min_batch = get_row_by_label(df_36_3, 'Minimum Batch Size (100 kg units)')
max_demand = {pid: int(float(row_max_demand[pid])) for pid in product_ids}
selling_price = {pid: float(row_selling_price[pid]) for pid in product_ids}
prod_cost = {pid: float(row_prod_cost[pid]) for pid in product_ids}
daily_quota = {pid: int(float(row_daily_quota[pid])) for pid in product_ids}
activation_cost = {pid: float(row_activation_cost[pid]) for pid in product_ids}
min_batch = {pid: int(float(row_min_batch[pid])) for pid in product_ids}
m = gp.Model('MonthlyProductionPlan')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(product_ids, vtype=gp.GRB.BINARY, name='')
profit_terms = ((selling_price[pid] - prod_cost[pid]) * x_vars[pid] - activation_cost[pid] * y_vars[pid] for pid in product_ids)
m.setObjective(gp.quicksum(profit_terms), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[pid] <= max_demand[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= 22 * daily_quota[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] >= min_batch[pid] * y_vars[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= max_demand[pid] * y_vars[pid] for pid in product_ids), name='')
m.addConstr(gp.quicksum((x_vars[pid] / daily_quota[pid] for pid in product_ids)) <= 22, name='TotalProductionDays')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('--- Production Plan ---')
    for pid in product_ids:
        x_val = x_vars[pid].X
        y_val = y_vars[pid].X
        if y_val > 0.5:
            print(f'{pid}: Produced {int(round(x_val))} (100kg units), Activated (y=1)')
        else:
            print(f'{pid}: Not produced (x=0), Not activated (y=0)')
else:
    print(f'No optimal solution found. Status: {m.status}')