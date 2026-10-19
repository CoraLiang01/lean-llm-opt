import gurobipy as gp
import pandas as pd
csv_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM12/Salesofsummerclothes.csv'
df = pd.read_csv(csv_path, sep=',')
product_names = df['Product Name'].tolist()
revenue = df['Revenue'].tolist()
demand = df['Demand'].tolist()
inventory = df['Initial Inventory'].tolist()
n = len(product_names)
m = gp.Model('MaximizeRevenue')
x = m.addVars(n, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in range(n))), gp.GRB.MAXIMIZE)
for i in range(n):
    m.addConstr(x[i] <= demand[i])
    m.addConstr(x[i] <= inventory[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Total Revenue: {m.objVal:.2f}')
    print('\nUnits sold per product:')
    for i in range(n):
        units = int(round(x[i].X))
        if units > 0:
            print(f'  {product_names[i]}: {units} units')
else:
    print(f'Optimization did not find an optimal solution. Status: {m.status}')