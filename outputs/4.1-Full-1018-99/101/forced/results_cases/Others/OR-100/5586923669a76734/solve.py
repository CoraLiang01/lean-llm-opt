import gurobipy as gp
import pandas as pd
import numpy as np
processing_time_unit_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/processing_time_unit.csv'
unit_price_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/unit_price.csv'
total_working_hours_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/total_working_hours.csv'
ptu_df = pd.read_csv(processing_time_unit_path, dtype=str, keep_default_na=False)
workshop_col = 'Unnamed: 0'
component_cols = [col for col in ptu_df.columns if col != workshop_col]
workshops = [row for row in ptu_df[workshop_col]]
components = component_cols.copy()
unit_price_df = pd.read_csv(unit_price_path, dtype=str, keep_default_na=False)
unit_price_df['unit_price'] = unit_price_df['unit_price'].astype(float)
unit_price_df = unit_price_df.set_index('Unnamed: 0', drop=True)
unit_price_dict = unit_price_df['unit_price'].to_dict()
twh_df = pd.read_csv(total_working_hours_path, dtype=str, keep_default_na=False)
twh_df['total_hours'] = twh_df['total_hours'].astype(float)

def norm_ws(s):
    return s.strip().casefold()
workshop_name_map = {norm_ws(w): w for w in workshops}
twh_df['workshop_norm'] = twh_df['workshop'].apply(norm_ws)
total_hours_dict = {}
for idx, row in twh_df.iterrows():
    ws_norm = row['workshop_norm']
    if ws_norm not in workshop_name_map:
        raise ValueError(f"Workshop '{row['workshop']}' in total_working_hours.csv not found in processing_time_unit.csv")
    ws_key = workshop_name_map[ws_norm]
    total_hours_dict[ws_key] = row['total_hours']
processing_time_unit = {}
for i, ws in enumerate(workshops):
    for comp in components:
        val = ptu_df.loc[i, comp]
        try:
            processing_time_unit[ws, comp] = float(val)
        except Exception:
            raise ValueError(f"Invalid processing time for workshop '{ws}', component '{comp}': '{val}'")
for comp in components:
    if comp not in unit_price_dict:
        raise ValueError(f"Component '{comp}' in processing_time_unit.csv not found in unit_price.csv")
for ws in workshops:
    if ws not in total_hours_dict:
        raise ValueError(f"Workshop '{ws}' in processing_time_unit.csv not found in total_working_hours.csv")
m = gp.Model('MaximizeTotalOutputValue')
x_vars = m.addVars(components, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((unit_price_dict[comp] * x_vars[comp] for comp in components)), gp.GRB.MAXIMIZE)
for ws in workshops:
    m.addConstr(gp.quicksum((processing_time_unit[ws, comp] * x_vars[comp] for comp in components)) <= total_hours_dict[ws], name=f"workshop_{ws.replace(' ', '_')}_capacity")
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total output value: {m.objVal:.2f}')
    print('\n--- Production Plan (component: quantity) ---')
    for comp in components:
        val = x_vars[comp].X
        if val > 1e-06:
            print(f'{comp}: {val:.4f}')
    print('\n--- Workshop Utilization (workshop: used/available hours) ---')
    for ws in workshops:
        used = sum((processing_time_unit[ws, comp] * x_vars[comp].X for comp in components))
        print(f'{ws}: {used:.2f} / {total_hours_dict[ws]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')