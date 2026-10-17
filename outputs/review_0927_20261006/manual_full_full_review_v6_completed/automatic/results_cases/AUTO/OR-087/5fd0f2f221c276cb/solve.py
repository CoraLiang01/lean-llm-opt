import gurobipy as gp
import pandas as pd
import numpy as np
file_36_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
file_36_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
file_36_3 = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df_36_1 = pd.read_csv(file_36_1, dtype=str, keep_default_na=False)
df_36_2 = pd.read_csv(file_36_2, dtype=str, keep_default_na=False)
df_36_3 = pd.read_csv(file_36_3, dtype=str, keep_default_na=False)
product_ids = [col for col in df_36_1.columns if col != 'Product']
if len(product_ids) != 80:
    raise ValueError(f'Expected 80 products, found {len(product_ids)}: {product_ids}')

def get_row_by_label(df, label):
    norm_label = label.strip().casefold()
    for idx, val in enumerate(df['Product']):
        if val.strip().casefold() == norm_label:
            return df.iloc[idx]
    raise KeyError(f"Row with label '{label}' not found in file.")
row_demand = get_row_by_label(df_36_1, 'Maximum Demand (100 kg units)')
row_price = get_row_by_label(df_36_1, 'Selling Price ($/100 kg)')
row_cost = get_row_by_label(df_36_1, 'Production Cost ($/100 kg)')
row_quota = get_row_by_label(df_36_1, 'Production Quota (max per day)')
max_demand = {pid: float(row_demand[pid]) for pid in product_ids}
selling_price = {pid: float(row_price[pid]) for pid in product_ids}
production_cost = {pid: float(row_cost[pid]) for pid in product_ids}
daily_quota = {pid: float(row_quota[pid]) for pid in product_ids}
row_activation = get_row_by_label(df_36_2, 'Activation Cost ($)')
activation_cost = {pid: float(row_activation[pid]) for pid in product_ids}
row_minbatch = get_row_by_label(df_36_3, 'Minimum Batch Size (100 kg units)')
min_batch = {pid: float(row_minbatch[pid]) for pid in product_ids}
m = gp.Model('MonthlyProductionPlan')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(product_ids, vtype=gp.GRB.BINARY, name='')
profit_terms = ((selling_price[pid] - production_cost[pid]) * x_vars[pid] - activation_cost[pid] * y_vars[pid] for pid in product_ids)
m.setObjective(gp.quicksum(profit_terms), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[pid] <= max_demand[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= 22 * daily_quota[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] >= min_batch[pid] * y_vars[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= max_demand[pid] * y_vars[pid] for pid in product_ids), name='')
m.addConstr(gp.quicksum((x_vars[pid] / daily_quota[pid] for pid in product_ids)) <= 22)
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