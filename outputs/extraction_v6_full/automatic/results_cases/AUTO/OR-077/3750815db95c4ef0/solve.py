import gurobipy as gp
import pandas as pd
import numpy as np
value_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(value_path, sep=',')
df['item'] = df['item'].astype(int)
df['value'] = df['value'].astype(float)
df['weight'] = df['weight'].astype(float)
items = df['item'].tolist()
value = dict(zip(df['item'], df['value']))
weight = dict(zip(df['item'], df['weight']))
m = gp.Model('KnapsackSelection')
x = m.addVars(items, vtype=gp.GRB.BINARY, name='x')
m.setObjective(gp.quicksum((value[i] * x[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in items)) <= 15, name='WeightLimit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Items ---')
    for i in items:
        if x[i].X > 0.5:
            print(f'Item {i}: value={value[i]}, weight={weight[i]}')
    total_weight = sum((weight[i] for i in items if x[i].X > 0.5))
    print(f'Total weight used: {total_weight:.2f} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')