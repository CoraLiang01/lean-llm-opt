import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv'
df = pd.read_csv(csv_path, sep=',')
organ_products = ['Organic Fruits', 'Organic Staples', 'Organic Vegetables']
organ_df = df[df['Sub Category'].astype(str).str.strip().isin(organ_products)].copy()
required_columns = ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in organ_df.columns:
        raise KeyError(f"Required column '{col}' not found in the data.")
missing_organs = set(organ_products) - set(organ_df['Sub Category'].astype(str).str.strip())
if missing_organs:
    raise ValueError(f'Missing Organ products in data: {missing_organs}')
organ_df['Sub Category'] = organ_df['Sub Category'].astype(str).str.strip()
organs = list(organ_df['Sub Category'])
revenue = organ_df.set_index('Sub Category')['Revenue'].to_dict()
demand = organ_df.set_index('Sub Category')['Demand'].to_dict()
inventory = organ_df.set_index('Sub Category')['Initial Inventory'].to_dict()
for i in organs:
    if pd.isnull(revenue[i]) or pd.isnull(demand[i]) or pd.isnull(inventory[i]):
        raise ValueError(f"Missing data for Organ product '{i}'.")
m = gp.Model('OrganProductRevenueMax')
x = m.addVars(organs, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in organs)), gp.GRB.MAXIMIZE)
m.addConstrs((x[i] <= demand[i] for i in organs), name='')
m.addConstrs((x[i] <= inventory[i] for i in organs), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Fulfillment Plan for Organ Products ---')
    for i in organs:
        print(f'{i}: Fulfilled units = {x[i].X:.2f} (Revenue per unit: {revenue[i]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')