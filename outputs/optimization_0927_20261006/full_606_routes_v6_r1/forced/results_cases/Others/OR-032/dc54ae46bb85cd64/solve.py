import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def is_books(row):
    return row['Product_Name'].strip().casefold().startswith('books')
books_df = df[df.apply(is_books, axis=1)].copy()
required_columns = ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in books_df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
books_df['Product_Name'] = books_df['Product_Name'].astype(str)
books_df.set_index('Product_Name', inplace=True)
for col in ['Revenue', 'Demand', 'Initial Inventory']:
    if col == 'Revenue':
        books_df[col] = books_df[col].astype(float)
    else:
        books_df[col] = books_df[col].astype(float)
books_products = list(books_df.index)
revenue = books_df['Revenue'].to_dict()
demand = books_df['Demand'].to_dict()
inventory = books_df['Initial Inventory'].to_dict()
for i in books_products:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f"Missing parameter for product '{i}'.")
m = gp.Model('BooksRevenueMaximization')
x_vars = m.addVars(books_products, lb=0.0, ub={i: min(demand[i], inventory[i]) for i in books_products}, vtype=gp.GRB.CONTINUOUS, name='')
m.addConstrs((x_vars[i] <= demand[i] for i in books_products), name='')
m.addConstrs((x_vars[i] <= inventory[i] for i in books_products), name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in books_products)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print("--- Fulfillment Plan for 'Books' Products ---")
    for i in books_products:
        print(f'Product: {i}')
        print(f'  Units fulfilled: {x_vars[i].X:.2f} (Demand: {demand[i]:.0f}, Inventory: {inventory[i]:.0f}, Revenue/unit: {revenue[i]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')