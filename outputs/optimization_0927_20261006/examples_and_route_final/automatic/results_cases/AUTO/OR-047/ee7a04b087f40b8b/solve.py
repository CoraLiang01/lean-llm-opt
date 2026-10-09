import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', dtype=str, keep_default_na=False)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', dtype=str, keep_default_na=False)
platform_ids = capacity_df['PlatformId'].astype(int).tolist()
genre_names = products_df['ProductName'].tolist()
capacity_dict = {}
for (idx, row) in capacity_df.iterrows():
    pid = int(row['PlatformId'])
    cap = int(row['Capacity'])
    capacity_dict[pid] = cap
value_dict = {}
weight_dict = {}
for (idx, row) in products_df.iterrows():
    genre = row['ProductName']
    try:
        value = int(row['Value'])
    except Exception:
        raise ValueError(f"Invalid Value for genre '{genre}' in products.csv")
    try:
        weight = int(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid Weight for genre '{genre}' in products.csv")
    value_dict[genre] = value
    weight_dict[genre] = weight
m = gp.Model('VideoGamePlatformListing')
x_vars = m.addVars(platform_ids, genre_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[genre] * x_vars[pid, genre] for pid in platform_ids for genre in genre_names)), gp.GRB.MAXIMIZE)
for pid in platform_ids:
    m.addConstr(gp.quicksum((weight_dict[genre] * x_vars[pid, genre] for genre in genre_names)) <= capacity_dict[pid], name=f'cap_{pid}')
m.optimize()