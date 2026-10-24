import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    path = 'Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv'
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV.")
    for col in ['Revenue', 'Demand', 'Initial Inventory']:
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
        except Exception as e:
            raise ValueError(f"Column '{col}' contains non-numeric values.") from e
    keys = df['Product Name'].tolist()
    r = dict(zip(keys, df['Revenue']))
    d = dict(zip(keys, df['Demand']))
    s = dict(zip(keys, df['Initial Inventory']))
    for key in keys:
        if key not in r or key not in d or key not in s:
            raise ValueError(f"Missing parameter for product '{key}'.")
    m = gurobipy.Model('RetailSalesOptimization')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.addConstrs((x[i] <= d[i] for i in keys), name='')
    m.addConstrs((x[i] <= s[i] for i in keys), name='')
    m.setObjective(gurobipy.quicksum((r[i] * x[i] for i in keys)), GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()