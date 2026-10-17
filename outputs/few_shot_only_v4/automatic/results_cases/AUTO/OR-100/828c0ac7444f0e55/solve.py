import gurobipy as gp
from gurobipy import GRB
import pandas as pd
processing_time_unit_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/processing_time_unit.csv'
unit_price_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/unit_price.csv'
total_working_hours_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/total_working_hours.csv'
unit_price_df = pd.read_csv(unit_price_path, dtype=str, keep_default_na=False)
if 'Component' not in unit_price_df.columns or 'UnitPrice' not in unit_price_df.columns:
    raise ValueError('unit_price.csv must have columns: Component, UnitPrice')
components = list(unit_price_df['Component'])
if len(components) != 111:
    raise ValueError('unit_price.csv must have 111 components')
unit_price = {row['Component']: float(row['UnitPrice']) for _, row in unit_price_df.iterrows()}
processing_time_df = pd.read_csv(processing_time_unit_path, dtype=str, keep_default_na=False)
if 'Component' not in processing_time_df.columns:
    raise ValueError('processing_time_unit.csv must have a Component column')
workshops = [col for col in processing_time_df.columns if col != 'Component']
if set(workshops) != set(['Casting', 'Milling', 'Finishing', 'Assembly', 'QA & Packaging']):
    raise ValueError('processing_time_unit.csv must have columns: Casting, Milling, Finishing, Assembly, QA & Packaging')
if sorted(list(processing_time_df['Component'])) != sorted(components):
    raise ValueError('Component identifiers in processing_time_unit.csv and unit_price.csv do not match')
processing_time = {}
for _, row in processing_time_df.iterrows():
    comp = row['Component']
    processing_time[comp] = {w: float(row[w]) for w in workshops}
total_working_hours_df = pd.read_csv(total_working_hours_path, dtype=str, keep_default_na=False)
if 'Workshop' not in total_working_hours_df.columns or 'TotalHours' not in total_working_hours_df.columns:
    raise ValueError('total_working_hours.csv must have columns: Workshop, TotalHours')
total_working_hours = {row['Workshop']: float(row['TotalHours']) for _, row in total_working_hours_df.iterrows()}
if set(total_working_hours.keys()) != set(workshops):
    raise ValueError('Workshop identifiers in total_working_hours.csv and processing_time_unit.csv do not match')
m = gp.Model('Factory_Production_Optimization')
x_vars = m.addVars(components, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((unit_price[i] * x_vars[i] for i in components)), GRB.MAXIMIZE)
for w in workshops:
    m.addConstr(gp.quicksum((processing_time[i][w] * x_vars[i] for i in components)) <= total_working_hours[w], name=f'time_{w}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')