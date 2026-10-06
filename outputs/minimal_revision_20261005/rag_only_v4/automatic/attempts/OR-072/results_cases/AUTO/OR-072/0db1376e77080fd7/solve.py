import pandas as pd
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',')
if not set(['Shift', 'Time', 'Number Required']).issubset(df.columns):
    raise ValueError('CSV missing required columns.')
periods = df['Shift'].astype(int).tolist()
if sorted(periods) != list(range(1, 25)):
    raise ValueError("Expected 24 consecutive periods indexed 1..24 in 'Shift' column.")
required = dict(zip(df['Shift'].astype(int), df['Number Required'].astype(int)))
model = gp.Model('BusStaffScheduling')
x = model.addVars(periods, vtype=GRB.INTEGER, lb=0, obj=0, name='')
for s in periods:
    covered_by = [(s - i - 1) % 24 + 1 for i in range(4)]
    model.addConstr(gp.quicksum((x[t] for t in covered_by)) >= required[s], name='')
model.setObjective(gp.quicksum((x[t] for t in periods)), GRB.MINIMIZE)
model.setParam('MIPGap', 0.0001)
model.optimize()
if model.Status == GRB.OPTIMAL:
    print(f'ObjVal: {model.ObjVal}')
    for t in periods:
        print(f'x[{t}]: {x[t].X}')
else:
    print(f'Solver status: {model.Status}')
m = model