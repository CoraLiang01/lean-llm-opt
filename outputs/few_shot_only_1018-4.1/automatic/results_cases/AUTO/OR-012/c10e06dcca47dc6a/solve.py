import re
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM3/OnlineSalesDataset.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()
header_match = re.match('^(.*?)\\n', csv_text)
if not header_match:
    raise ValueError('CSV header not found.')
header = header_match.group(1).strip().split(',')
rows = re.findall('^(?!Product Name)(.+)$', csv_text, flags=re.MULTILINE)
if not rows:
    raise ValueError('No data rows found in CSV.')
data = []
for row in rows:
    fields = re.split('\\s{2,}|\\t|,(?=(?:[^"]*"[^"]*")*[^"]*$)', row.strip())
    fields = [f.strip() for f in fields if f.strip() != '']
    if len(fields) != len(header):
        raise ValueError(f'Row does not match header length: {row}')
    data.append(fields)
df = pd.DataFrame(data, columns=header)
for col in ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']:
    if col not in df.columns:
        raise ValueError(f'Missing required column: {col}')
products = df['Product Name'].tolist()

def parse_float(x):
    try:
        return float(x)
    except Exception:
        raise ValueError(f'Cannot parse float: {x}')

def parse_int(x):
    try:
        return int(float(x))
    except Exception:
        raise ValueError(f'Cannot parse int: {x}')
revenue = {}
demand = {}
inventory = {}
for idx, row in df.iterrows():
    i = row['Product Name']
    try:
        revenue[i] = parse_float(row['Revenue'])
        demand[i] = parse_int(row['Demand'])
        inventory[i] = parse_int(row['Initial Inventory'])
    except Exception as e:
        raise ValueError(f"Error parsing row for product '{i}': {e}")
for i in products:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f'Missing parameter for product: {i}')

def build_model(products, revenue, demand, inventory):
    m = gp.Model('OnlineSalesRevenueMax')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
    m.addConstrs((x_vars[i] <= demand[i] for i in products), name='')
    m.addConstrs((x_vars[i] <= inventory[i] for i in products), name='')
    return m
m = build_model(products, revenue, demand, inventory)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')