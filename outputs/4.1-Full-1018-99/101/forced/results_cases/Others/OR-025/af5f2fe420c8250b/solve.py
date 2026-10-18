import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

def is_tablet(name):
    return name.strip().casefold().startswith('tablet')
tablet_mask = df['Product Name'].apply(is_tablet)
tablets_df = df.loc[tablet_mask].copy()
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in tablets_df.columns:
        raise KeyError(f"Required column '{col}' not found in the data.")
tablet_ids = tablets_df['Product Name'].tolist()

def to_float(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to float: {e}")

def to_int(series, colname):
    try:
        return series.astype(int)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to int: {e}")
revenue = dict(zip(tablet_ids, to_float(tablets_df['Revenue'], 'Revenue')))
demand = dict(zip(tablet_ids, to_int(tablets_df['Demand'], 'Demand')))
initial_inventory = dict(zip(tablet_ids, to_int(tablets_df['Initial Inventory'], 'Initial Inventory')))
fulfillment_ub = {i: min(demand[i], initial_inventory[i]) for i in tablet_ids}
m = gp.Model('TabletRevenueMaximization')
x_vars = m.addVars(tablet_ids, vtype=gp.GRB.INTEGER, lb=0, ub=fulfillment_ub, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in tablet_ids)), gp.GRB.MAXIMIZE)
for i in tablet_ids:
    m.addConstr(x_vars[i] <= demand[i], name=f'demand_{i}')
    m.addConstr(x_vars[i] <= initial_inventory[i], name=f'inventory_{i}')
    m.addConstr(x_vars[i] >= 0, name=f'nonneg_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Fulfillment Plan for Tablets ---')
    for i in tablet_ids:
        fulfilled = int(round(x_vars[i].X))
        if fulfilled > 0:
            print(f'Tablet Model: {i} | Fulfilled: {fulfilled} | Revenue/unit: {revenue[i]:.2f} | Total Revenue: {revenue[i] * fulfilled:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')