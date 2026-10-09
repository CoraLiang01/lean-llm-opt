import gurobipy as gp
import pandas as pd
import numpy as np
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', dtype=str, keep_default_na=False)
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv', dtype=str, keep_default_na=False)
if not {'ProductName', 'Weight', 'Value'}.issubset(products_df.columns):
    raise KeyError('products.csv must contain columns: ProductName, Weight, Value')
if 'Capacity' not in capacity_df.columns:
    raise KeyError('capacity.csv must contain column: Capacity')
product_ids = products_df['ProductName'].tolist()
try:
    weight_param = {pid: int(products_df.loc[products_df['ProductName'] == pid, 'Weight'].values[0]) for pid in product_ids}
    value_param = {pid: int(products_df.loc[products_df['ProductName'] == pid, 'Value'].values[0]) for pid in product_ids}
except Exception as e:
    raise ValueError(f'Error converting Weight/Value to int for products: {e}')
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row for total capacity.')
try:
    total_capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Error converting Capacity to int: {e}')
m = gp.Model('SupermarketStockReplenishment')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_param[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_param[pid] * x_vars[pid] for pid in product_ids)) <= total_capacity, name='StockCapacity')
m.optimize()