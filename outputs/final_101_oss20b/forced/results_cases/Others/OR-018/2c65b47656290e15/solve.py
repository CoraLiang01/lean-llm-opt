import pandas as pd
import gurobipy as gp
data_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv'
df = pd.read_csv(data_path)
df_baby = df[df['Product Name'].str.startswith('Baby')]
product_names = df_baby['Product Name'].tolist()
revenue_list = df_baby['Revenue'].tolist()
demand_list = df_baby['Demand'].tolist()
inventory_list = df_baby['Initial Inventory'].tolist()
m = gp.Model('BabyProductRevenueMaximization')
x = m.addVars(len(product_names), lb=0.0)
obj_expr = gp.quicksum((revenue_list[i] * x[i] for i in range(len(product_names))))
m.setObjective(obj_expr, gp.GRB.MAXIMIZE)
for i in range(len(product_names)):
    m.addConstr(x[i] <= demand_list[i])
    m.addConstr(x[i] <= inventory_list[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('\n--- Sales Plan ---')
    for i in range(len(product_names)):
        units = x[i].X
        if units > 1e-06:
            print(f'{product_names[i]}: {units:.0f} units')
else:
    print(f'No optimal solution found. Status: {m.status}')