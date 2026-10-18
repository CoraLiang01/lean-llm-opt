import pandas as pd
import gurobipy as gp
from gurobipy import GRB
processing_time_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/processing_time_unit.csv', dtype=str, keep_default_na=False)
unit_price_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/unit_price.csv', dtype=str, keep_default_na=False)
total_working_hours_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/total_working_hours.csv', dtype=str, keep_default_na=False)
component_ids = [col for col in processing_time_df.columns if col != 'Unnamed: 0']
workshop_ids = list(processing_time_df['Unnamed: 0'])
processing_time = {}
for w in workshop_ids:
    row = processing_time_df[processing_time_df['Unnamed: 0'] == w]
    if row.empty:
        raise ValueError(f'Workshop {w} not found in processing_time_unit.csv')
    processing_time[w] = {}
    for c in component_ids:
        val = row.iloc[0][c]
        try:
            processing_time[w][c] = float(val)
        except Exception:
            raise ValueError(f'Invalid processing time for workshop {w}, component {c}: {val}')
unit_price = {}
for idx, row in unit_price_df.iterrows():
    c = row['Unnamed: 0']
    if c not in component_ids:
        raise ValueError(f'Component {c} in unit_price.csv not found in processing_time_unit.csv')
    try:
        unit_price[c] = float(row['unit_price'])
    except Exception:
        raise ValueError(f"Invalid unit price for component {c}: {row['unit_price']}")
total_hours = {}
for idx, row in total_working_hours_df.iterrows():
    w = row['workshop']
    if w not in workshop_ids:
        raise ValueError(f'Workshop {w} in total_working_hours.csv not found in processing_time_unit.csv')
    try:
        total_hours[w] = float(row['total_hours'])
    except Exception:
        raise ValueError(f"Invalid total hours for workshop {w}: {row['total_hours']}")
if set(unit_price.keys()) != set(component_ids):
    raise ValueError('Mismatch between components in unit_price.csv and processing_time_unit.csv')
if set(total_hours.keys()) != set(workshop_ids):
    raise ValueError('Mismatch between workshops in total_working_hours.csv and processing_time_unit.csv')
m = gp.Model('factory_production')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(component_ids, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((unit_price[c] * x_vars[c] for c in component_ids)), GRB.MAXIMIZE)
for w in workshop_ids:
    m.addConstr(gp.quicksum((processing_time[w][c] * x_vars[c] for c in component_ids)) <= total_hours[w], name=f'time_{w}')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')