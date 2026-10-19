import gurobipy as gp
import pandas as pd
df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv', sep=',')
mask = df['Product Name'].str.contains('FDK57', case=False, na=False)
df_fdk57 = df[mask].copy()
products = df_fdk57['Product Name'].tolist()
revenue = dict(zip(products, df_fdk57['Revenue']))
demand = dict(zip(products, df_fdk57['Demand']))
inventory = dict(zip(products, df_fdk57['Initial Inventory']))
m = gp.Model('FDK57_Sales_Optimization')
x = m.addVars(products, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), gp.GRB.MAXIMIZE)
m.addConstrs((x[i] <= inventory[i] for i in products))
m.addConstrs((x[i] <= demand[i] for i in products))
m.addConstrs((x[i] >= 0 for i in products))
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('\nOptimal sales quantities:')
    for i in products:
        qty = x[i].X
        if qty > 0:
            print(f'{i}: {int(qty)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')