import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture2/41.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Product Name' not in df.columns:
    raise KeyError("Missing required column: 'Product Name'")
product_names = df['Product Name'].tolist()

def to_float_series(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' contains non-numeric values.") from e

def to_int_series(series, colname):
    try:
        return series.astype(int)
    except Exception as e:
        raise ValueError(f"Column '{colname}' contains non-integer values.") from e
labor_per_unit = to_float_series(df['Labor per unit'], 'Labor per unit')
material_per_unit = to_float_series(df['Material per unit'], 'Material per unit')
selling_price = to_int_series(df['Selling Price'], 'Selling Price')
variable_cost = to_int_series(df['Variable Cost'], 'Variable Cost')
labor_per_unit_dict = dict(zip(product_names, labor_per_unit))
material_per_unit_dict = dict(zip(product_names, material_per_unit))
selling_price_dict = dict(zip(product_names, selling_price))
variable_cost_dict = dict(zip(product_names, variable_cost))
LABOR_CAPACITY = 1650.0
MATERIAL_CAPACITY = 1850.0
FIXED_WEEKLY_COST = 4500.0
m = gp.Model('RedBeanClothingFactory')
x_vars = m.addVars(product_names, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
profit_coeffs = {p: selling_price_dict[p] - variable_cost_dict[p] for p in product_names}
objective_expr = gp.quicksum((profit_coeffs[p] * x_vars[p] for p in product_names)) - FIXED_WEEKLY_COST
m.setObjective(objective_expr, gp.GRB.MAXIMIZE)
labor_expr = gp.quicksum((labor_per_unit_dict[p] * x_vars[p] for p in product_names))
m.addConstr(labor_expr <= LABOR_CAPACITY, name='LaborCapacity')
material_expr = gp.quicksum((material_per_unit_dict[p] * x_vars[p] for p in product_names))
m.addConstr(material_expr <= MATERIAL_CAPACITY, name='MaterialCapacity')
m.optimize()