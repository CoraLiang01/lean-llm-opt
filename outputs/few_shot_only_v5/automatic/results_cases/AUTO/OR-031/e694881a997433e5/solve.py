import re
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()
header_match = re.search('^\\s*Full_Product_Name\\s+Revenue\\s+Demand\\s+Initial Inventory\\s*$', csv_text, re.MULTILINE)
if not header_match:
    raise ValueError('CSV header not found or does not match required columns.')
row_pattern = re.compile('^\\s*(.+?)\\s+([0-9.]+)\\s+([0-9]+)\\s+([0-9]+)\\s*$', re.MULTILINE)
rows = row_pattern.findall(csv_text)
if not rows:
    raise ValueError('No data rows matched in CSV.')
df = pd.DataFrame(rows, columns=['Full_Product_Name', 'Revenue', 'Demand', 'Initial Inventory'])
df['Revenue'] = df['Revenue'].astype(float)
df['Demand'] = df['Demand'].astype(int)
df['Initial Inventory'] = df['Initial Inventory'].astype(int)
products = df['Full_Product_Name'].tolist()
revenue = dict(zip(df['Full_Product_Name'], df['Revenue']))
demand = dict(zip(df['Full_Product_Name'], df['Demand']))
inventory = dict(zip(df['Full_Product_Name'], df['Initial Inventory']))
if set(products) != set(revenue.keys()) or set(products) != set(demand.keys()) or set(products) != set(inventory.keys()):
    raise ValueError('Mismatch in product indices among parameters.')
m = gp.Model('DairyGoodsOrderFulfillment')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= inventory[i] for i in products), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')