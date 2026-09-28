import gurobipy as gp
import pandas as pd
import numpy as np
import re
path_processing_time = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/processing_time_unit.csv'
path_unit_price = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/unit_price.csv'
path_total_hours = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/total_working_hours.csv'
df_proc = pd.read_csv(path_processing_time, sep=',')
df_price = pd.read_csv(path_unit_price, sep=',')
df_hours = pd.read_csv(path_total_hours, sep=',')
components = df_price['Unnamed: 0'].astype(str).tolist()
workshops = df_hours['workshop'].astype(str).tolist()
unit_price = dict(zip(df_price['Unnamed: 0'].astype(str), df_price['unit_price']))
total_hours = dict(zip(df_hours['workshop'].astype(str), df_hours['total_hours']))
df_proc = df_proc.set_index('Unnamed: 0')
missing_workshops = set(workshops) - set(df_proc.index)
if missing_workshops:
    raise ValueError(f'Missing workshops in processing_time_unit.csv: {missing_workshops}')
missing_components = set(components) - set(df_proc.columns)
if missing_components:
    raise ValueError(f'Missing components in processing_time_unit.csv: {missing_components}')
processing_time_unit = {}
for w in workshops:
    for c in components:
        val = df_proc.loc[w, c]
        if pd.isnull(val):
            raise ValueError(f"Missing processing time for workshop '{w}', component '{c}'")
        processing_time_unit[w, c] = float(val)
m = gp.Model('MaximizeOutputValue')
x = m.addVars(components, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((unit_price[c] * x[c] for c in components)), gp.GRB.MAXIMIZE)
for w in workshops:
    m.addConstr(gp.quicksum((processing_time_unit[w, c] * x[c] for c in components)) <= total_hours[w], name=f"time_{w.replace(' ', '_')}")
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total output value: {m.objVal:.2f}')
    print('\n--- Production Plan (nonzero quantities) ---')
    for c in components:
        val = x[c].X
        if val > 1e-06:
            print(f'Component {c}: {val:.4f} units (unit price: {unit_price[c]})')
    print('\n--- Workshop Utilization ---')
    for w in workshops:
        used = sum((processing_time_unit[w, c] * x[c].X for c in components))
        print(f'{w}: {used:.2f} / {total_hours[w]} hours used ({used / total_hours[w] * 100:.2f}%)')
else:
    print(f'No optimal solution found. Status: {m.status}')