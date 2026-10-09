import gurobipy as gp
import pandas as pd
import numpy as np
processing_time_unit_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/processing_time_unit.csv'
unit_price_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/unit_price.csv'
total_working_hours_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/total_working_hours.csv'
ptu_df = pd.read_csv(processing_time_unit_path, dtype=str, keep_default_na=False)
workshop_col = 'Unnamed: 0'
component_ids = [col for col in ptu_df.columns if col != workshop_col]
workshop_ids = ptu_df[workshop_col].tolist()
unit_price_df = pd.read_csv(unit_price_path, dtype=str, keep_default_na=False)
unit_price_df['unit_price'] = unit_price_df['unit_price'].astype(float)
unit_price_df.set_index('Unnamed: 0', inplace=True)
unit_price_dict = unit_price_df['unit_price'].to_dict()
twh_df = pd.read_csv(total_working_hours_path, dtype=str, keep_default_na=False)
twh_df['total_hours'] = twh_df['total_hours'].astype(float)
workshop_name_map = {w: w for w in workshop_ids}

def norm_ws(s):
    return s.strip().casefold()
workshop_norm_to_id = {norm_ws(w): w for w in workshop_ids}
twh_dict = {}
for (idx, row) in twh_df.iterrows():
    ws_norm = norm_ws(row['workshop'])
    if ws_norm not in workshop_norm_to_id:
        raise KeyError(f"Workshop '{row['workshop']}' in total_working_hours.csv not found in processing_time_unit.csv")
    ws_id = workshop_norm_to_id[ws_norm]
    twh_dict[ws_id] = float(row['total_hours'])
processing_time_unit = {}
for (i, ws) in enumerate(workshop_ids):
    processing_time_unit[ws] = {}
    for comp in component_ids:
        val = ptu_df.loc[i, comp]
        try:
            processing_time_unit[ws][comp] = float(val)
        except Exception:
            raise ValueError(f"Invalid processing time for workshop '{ws}', component '{comp}': '{val}'")
for comp in component_ids:
    if comp not in unit_price_dict:
        raise KeyError(f"Component '{comp}' missing from unit_price.csv")
for ws in workshop_ids:
    if ws not in twh_dict:
        raise KeyError(f"Workshop '{ws}' missing from total_working_hours.csv")
m = gp.Model('MaximizeTotalOutputValue')
x_vars = m.addVars(component_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((unit_price_dict[comp] * x_vars[comp] for comp in component_ids)), gp.GRB.MAXIMIZE)
for ws in workshop_ids:
    m.addConstr(gp.quicksum((processing_time_unit[ws][comp] * x_vars[comp] for comp in component_ids)) <= twh_dict[ws], name=f'workshop_{ws}_hours')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total output value: {m.objVal:.2f}')
    print('--- Production Plan (component: quantity) ---')
    for comp in component_ids:
        qty = x_vars[comp].X
        if qty > 1e-06:
            print(f'{comp}: {qty:.2f}')
    print('--- Workshop Utilization ---')
    for ws in workshop_ids:
        used = sum((processing_time_unit[ws][comp] * x_vars[comp].X for comp in component_ids))
        print(f'{ws}: {used:.2f} / {twh_dict[ws]:.2f} hours used')
else:
    print(f'No optimal solution found. Status: {m.status}')