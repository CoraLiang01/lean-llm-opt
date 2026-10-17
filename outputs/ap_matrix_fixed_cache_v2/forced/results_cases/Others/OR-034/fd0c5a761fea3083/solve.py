import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM25/Frenchbakerydailysales.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
product_ids = df['Product Name'].tolist()

def safe_float(x):
    try:
        return float(x)
    except Exception:
        raise ValueError(f"Could not convert value '{x}' to float.")

def safe_int(x):
    try:
        return int(float(x))
    except Exception:
        raise ValueError(f"Could not convert value '{x}' to int.")
revenue = {pid: safe_float(val) for pid, val in zip(product_ids, df['Revenue'])}
demand = {pid: safe_int(val) for pid, val in zip(product_ids, df['Demand'])}
init_inventory = {pid: safe_int(val) for pid, val in zip(product_ids, df['Initial Inventory'])}
fulfill_upper = {pid: min(demand[pid], init_inventory[pid]) for pid in product_ids}
m = gp.Model('FrenchBakeryRevenueMaximization')
x_vars = m.addVars(product_ids, lb=0.0, ub=[fulfill_upper[pid] for pid in product_ids], vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
for pid in product_ids:
    m.addConstr(x_vars[pid] <= demand[pid], name=f'demand_{pid}')
    m.addConstr(x_vars[pid] <= init_inventory[pid], name=f'inventory_{pid}')
    m.addConstr(x_vars[pid] >= 0, name=f'nonneg_{pid}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Fulfillment Plan ---')
    for pid in product_ids:
        qty = x_vars[pid].X
        if qty > 1e-06:
            print(f'{pid}: fulfill {qty:.2f} units (Revenue per unit: {revenue[pid]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')