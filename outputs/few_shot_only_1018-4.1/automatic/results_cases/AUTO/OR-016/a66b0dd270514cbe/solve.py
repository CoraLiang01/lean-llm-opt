import re
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv'
with open(csv_path, 'r') as f:
    csv_text = f.read()
header_match = re.search('^\\s*Product Name\\s+Revenue\\s+Demand\\s+Initial Inventory\\s*$', csv_text, re.MULTILINE)
if not header_match:
    raise ValueError('CSV header not matched exactly as required.')
row_pattern = re.compile('^\\s*(.+?)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s*$', re.MULTILINE)
rows = row_pattern.findall(csv_text)
if not rows or len(rows) == 0:
    raise ValueError('No data rows matched in CSV.')
df = pd.DataFrame(rows, columns=['Product Name', 'Revenue', 'Demand', 'Initial Inventory'])
df['Revenue'] = df['Revenue'].astype(float)
df['Demand'] = df['Demand'].astype(float)
df['Initial Inventory'] = df['Initial Inventory'].astype(float)
products = df['Product Name'].tolist()
revenue = dict(zip(df['Product Name'], df['Revenue']))
demand = dict(zip(df['Product Name'], df['Demand']))
inventory = dict(zip(df['Product Name'], df['Initial Inventory']))
if set(products) != set(revenue.keys()) or set(products) != set(demand.keys()) or set(products) != set(inventory.keys()):
    raise ValueError('Mismatch in product indices and parameter keys.')
m = gp.Model('Retail_Revenue_Allocation')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in products), name='')
m.addConstrs((x_vars[i] <= inventory[i] for i in products), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')