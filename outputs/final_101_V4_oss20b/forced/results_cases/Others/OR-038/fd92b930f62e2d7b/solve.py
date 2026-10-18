import gurobipy as gp
import pandas as pd
capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv'
products_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
capacity_df['VehicleType'] = capacity_df['VehicleType'].str.strip()
products_df['ProductName'] = products_df['ProductName'].str.strip()
merged_df = pd.merge(capacity_df, products_df, left_on='VehicleType', right_on='ProductName', how='inner')
if len(merged_df) != len(capacity_df):
    raise ValueError('Mismatch between capacity and products data')
V = merged_df['VehicleID'].tolist()
c = dict(zip(merged_df['VehicleID'], merged_df['Capacity']))
b = dict(zip(merged_df['VehicleID'], merged_df['Value']))
vehicle_type_map = dict(zip(merged_df['VehicleID'], merged_df['VehicleType']))
C = sum(c.values())
m = gp.Model('NewCarSalesInventory')
x = m.addVars(V, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((b[i] * x[i] for i in V)), gp.GRB.MAXIMIZE)
for i in V:
    m.addConstr(x[i] <= c[i])
m.addConstr(gp.quicksum((x[i] for i in V)) <= C)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total benefit: {m.objVal:.2f}')
    print('--- Daily Ordering Plan ---')
    for i in V:
        qty = x[i].X
        if qty > 0:
            print(f'VehicleID {i} ({vehicle_type_map[i]}): {int(qty)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')