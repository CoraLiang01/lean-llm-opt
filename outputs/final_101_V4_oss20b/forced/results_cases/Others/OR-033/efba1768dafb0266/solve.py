import gurobipy as gp
import pandas as pd
csv_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv'
df = pd.read_csv(csv_path)
baby_df = df[df['Product Name'].str.contains('Baby', case=False, na=False)].copy()
baby_df['Revenue'] = baby_df['Revenue'].astype(float)
baby_df['Demand'] = baby_df['Demand'].astype(int)
baby_df['Initial Inventory'] = baby_df['Initial Inventory'].astype(int)
m = gp.Model('BabyRevenueMaximization')
x = {}
for idx, prod in enumerate(baby_df['Product Name']):
    var = m.addVar(vtype=gp.GRB.INTEGER, lb=0)
    x[prod] = var
for idx, prod in enumerate(baby_df['Product Name']):
    demand = baby_df.loc[baby_df['Product Name'] == prod, 'Demand'].iloc[0]
    inventory = baby_df.loc[baby_df['Product Name'] == prod, 'Initial Inventory'].iloc[0]
    m.addConstr(x[prod] <= demand)
    m.addConstr(x[prod] <= inventory)
obj = gp.quicksum((baby_df.loc[baby_df['Product Name'] == prod, 'Revenue'].iloc[0] * x[prod] for prod in baby_df['Product Name']))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    for prod in baby_df['Product Name']:
        print(f'{prod}: {x[prod].X} units sold')
else:
    print(f'No optimal solution found. Status: {m.status}')