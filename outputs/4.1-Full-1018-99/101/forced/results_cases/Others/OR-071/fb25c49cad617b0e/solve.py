import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture2/41.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Product Name' not in df.columns:
    raise KeyError("Missing required column: 'Product Name'")
products = df['Product Name'].tolist()
required_columns = ['Labor per unit', 'Material per unit', 'Selling Price', 'Variable Cost']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Missing required column: '{col}'")
    if (df[col].str.strip() == '').any():
        raise ValueError(f"Blank or missing entries found in column '{col}'")
    try:
        df[col] = pd.to_numeric(df[col], errors='raise')
    except Exception as e:
        raise ValueError(f"Failed to convert column '{col}' to numeric: {e}")
labor_per_unit = dict(zip(df['Product Name'], df['Labor per unit']))
material_per_unit = dict(zip(df['Product Name'], df['Material per unit']))
selling_price = dict(zip(df['Product Name'], df['Selling Price']))
variable_cost = dict(zip(df['Product Name'], df['Variable Cost']))
for p in products:
    for param, d in [('Labor per unit', labor_per_unit), ('Material per unit', material_per_unit), ('Selling Price', selling_price), ('Variable Cost', variable_cost)]:
        if p not in d:
            raise KeyError(f"Product '{p}' missing parameter '{param}'")
labor_capacity = 1650.0
material_capacity = 1850.0
fixed_weekly_operating_cost = 4500.0
m = gp.Model('RedBeanClothingFactory')
x_vars = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
profit_terms = [(selling_price[p] - variable_cost[p]) * x_vars[p] for p in products]
m.setObjective(gp.quicksum(profit_terms) - fixed_weekly_operating_cost, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_per_unit[p] * x_vars[p] for p in products)) <= labor_capacity, name='labor_capacity')
m.addConstr(gp.quicksum((material_per_unit[p] * x_vars[p] for p in products)) <= material_capacity, name='material_capacity')
m.optimize()