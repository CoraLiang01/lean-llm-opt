import pandas as pd
import gurobipy
from gurobipy import GRB
csv_path = 'Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
row = df.loc[df['id_number'] == 'id999']
if row.empty:
    raise ValueError("No record found for id_number 'id999'.")
revenue = float(row.iloc[0]['Revenue'])
demand = int(row.iloc[0]['Demand'])
inventory = int(row.iloc[0]['Initial Inventory'])
I_set = [999]
R = {999: revenue}
D = {999: demand}
I = {999: inventory}
m = gurobipy.Model()
m.setParam('MIPGap', 1e-06)
x = m.addVars(I_set, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gurobipy.quicksum((R[i] * x[i] for i in I_set)), GRB.MAXIMIZE)
m.addConstr(x[999] <= D[999], name='demand')
m.addConstr(x[999] <= I[999], name='inventory')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')