import gurobipy as gp
import pandas as pd
import re
path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv'
df = pd.read_csv(path, sep=',')
tablet_mask = df['Product Name'].str.contains('TABLET', case=False, na=False)
tablet_df = df[tablet_mask].copy()
I = tablet_df['Product Name'].tolist()
revenue = dict(zip(I, tablet_df['Revenue']))
demand = dict(zip(I, tablet_df['Demand']))
initial_inventory = dict(zip(I, tablet_df['Initial Inventory']))

def sanitize(name):
    return re.sub('[^A-Za-z0-9_]', '_', name)
m = gp.Model('TabletRevenueMaximization')
x = m.addVars(I, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in I)), gp.GRB.MAXIMIZE)
for i in I:
    m.addConstr(x[i] <= initial_inventory[i])
    m.addConstr(x[i] <= demand[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_rev = m.objVal
    print(f'Total maximum revenue: {total_rev:.2f}')
    print('Sales plan:')
    for i in I:
        qty = x[i].X
        if qty > 0:
            print(f'  {i}: {int(qty)} units')
else:
    print(f'Optimization ended with status {m.status}')