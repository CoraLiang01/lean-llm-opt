import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv'
try:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
except Exception as e:
    raise RuntimeError(f'Failed to read CSV: {e}')
product_regex = re.compile('^TABLET_')
I = []
for idx, row in df.iterrows():
    pname = row['Product Name'].strip()
    if product_regex.match(pname):
        I.append(pname)
if not I:
    raise ValueError('No TABLET_ products found in the CSV.')
A_i = {}
d_i = {}
s_i = {}
for pname in I:
    row = df[df['Product Name'].str.strip() == pname]
    if row.empty:
        raise ValueError(f'Product {pname} not found in CSV after filtering.')
    try:
        revenue = float(row.iloc[0]['Revenue'])
        demand = int(float(row.iloc[0]['Demand']))
        inventory = int(float(row.iloc[0]['Initial Inventory']))
    except Exception as e:
        raise ValueError(f'Failed to parse numeric fields for {pname}: {e}')
    A_i[pname] = revenue
    d_i[pname] = demand
    s_i[pname] = inventory
if set(I) != set(A_i.keys()) or set(I) != set(d_i.keys()) or set(I) != set(s_i.keys()):
    raise ValueError('Mismatch in index set and parameter keys.')
m = gp.Model('Tablet_Fulfillment')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((A_i[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= s_i[i] for i in I), name='')
m.addConstrs((x_vars[i] <= d_i[i] for i in I), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')