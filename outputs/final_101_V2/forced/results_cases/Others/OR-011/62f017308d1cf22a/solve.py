import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv'
df = pd.read_csv(csv_path, sep=',')
id999_mask = df['id_number'].astype(str) == 'id999'
df_id999 = df.loc[id999_mask].copy()
if df_id999.shape[0] == 0:
    raise ValueError("No products with id_number == 'id999' found in the dataset.")
products = df_id999['id_number'].astype(str).tolist()
if len(products) != len(df_id999):
    raise ValueError('Mismatch in product identifier extraction.')
revenue = df_id999['Revenue'].astype(float).tolist()
demand = df_id999['Demand'].astype(int).tolist()
init_inventory = df_id999['Initial Inventory'].astype(int).tolist()
revenue_dict = {pid: rev for pid, rev in zip(products, revenue)}
demand_dict = {pid: d for pid, d in zip(products, demand)}
inventory_dict = {pid: inv for pid, inv in zip(products, init_inventory)}
m = gp.Model('MaximizeRevenue_id999')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue_dict[i] * x[i] for i in products)), gp.GRB.MAXIMIZE)
m.addConstrs((x[i] <= inventory_dict[i] for i in products), name='')
m.addConstrs((x[i] <= demand_dict[i] for i in products), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Fulfillment Plan for id999 Products ---')
    for i in products:
        print(f'Product {i}:')
        print(f'  Fulfilled units (x): {int(round(x[i].X))}')
        print(f'  Revenue per unit: {revenue_dict[i]:.2f}')
        print(f'  Demand: {demand_dict[i]}')
        print(f'  Initial Inventory: {inventory_dict[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')