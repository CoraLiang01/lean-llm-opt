import gurobipy as gp
import pandas as pd
path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(path, sep=',')
value_dict = dict(zip(df['item'], df['value']))
weight_dict = dict(zip(df['item'], df['weight']))
m = gp.Model('Knapsack')
x = m.addVars(df['item'], vtype=gp.GRB.BINARY)
m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in df['item'])), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in df['item'])) <= 15)
m.optimize()
print(f'Optimal total value: {m.objVal:.0f}')
selected_items = [i for i in df['item'] if x[i].X > 0.5]
total_weight_used = sum((weight_dict[i] for i in selected_items))
print(f'Total weight used: {total_weight_used}')
print('Selected items:', selected_items)