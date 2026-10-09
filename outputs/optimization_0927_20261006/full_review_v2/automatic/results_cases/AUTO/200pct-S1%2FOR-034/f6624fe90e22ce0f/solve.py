import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def norm(s):
    return re.sub('\\s+', '', s.strip().casefold())
colmap = {norm(col): col for col in df.columns}
item_col = colmap[norm('item')]
value_col = colmap[norm('value')]
weight_col = colmap[norm('weight')]
items = df[item_col].astype(str).tolist()
value_dict = {}
weight_dict = {}
for (idx, row) in df.iterrows():
    item_id = str(row[item_col])
    try:
        value = int(row[value_col])
        weight = int(row[weight_col])
    except Exception as e:
        raise ValueError(f'Non-integer value or weight for item {item_id}: {e}')
    value_dict[item_id] = value
    weight_dict[item_id] = weight
m = gp.Model('KnapsackDisplay')
x_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in items)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Items ---')
    for i in items:
        if x_vars[i].X > 0.5:
            print(f'Item {i}: value={value_dict[i]}, weight={weight_dict[i]}')
    total_weight = sum((weight_dict[i] for i in items if x_vars[i].X > 0.5))
    print(f'Total weight used: {total_weight} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')