import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def is_books(row):
    return row['Product_Name'].strip().casefold().startswith('books')
books_df = df[df.apply(is_books, axis=1)].copy()
required_cols = ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in books_df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
books_df['Product_Name'] = books_df['Product_Name'].astype(str)
books_df = books_df.set_index('Product_Name', drop=False)
for col in ['Revenue', 'Demand', 'Initial Inventory']:
    if col == 'Revenue':
        books_df[col] = books_df[col].astype(float)
    else:
        books_df[col] = pd.to_numeric(books_df[col], errors='raise')
books_products = list(books_df.index)
revenue = books_df['Revenue'].to_dict()
demand = books_df['Demand'].to_dict()
inventory = books_df['Initial Inventory'].to_dict()
fulfill_ub = {i: min(int(demand[i]), int(inventory[i])) for i in books_products}
m = gp.Model('BooksRevenueMaximization')
x_vars = m.addVars(books_products, vtype=gp.GRB.INTEGER, lb=0, ub=[fulfill_ub[i] for i in books_products], name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in books_products)), gp.GRB.MAXIMIZE)
for i in books_products:
    m.addConstr(x_vars[i] <= demand[i], name=f'demand_{i}')
    m.addConstr(x_vars[i] <= inventory[i], name=f'inventory_{i}')
    m.addConstr(x_vars[i] >= 0, name=f'nonneg_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print("--- Fulfillment Plan for 'Books' Products ---")
    for i in books_products:
        print(f'{i}: Fulfilled {int(round(x_vars[i].X))} units (Demand: {int(demand[i])}, Inventory: {int(inventory[i])}, Revenue/unit: {revenue[i]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')