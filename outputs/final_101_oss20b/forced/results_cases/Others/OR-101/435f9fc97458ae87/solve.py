import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
device_time_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/device_time.csv'
capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/monthly_device_capacity.csv'
profit_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/unit_product_profits.csv'
df_time = pd.read_csv(device_time_path, sep=',')
df_cap = pd.read_csv(capacity_path, sep=',')
df_profit = pd.read_csv(profit_path, sep=',')
df_time.set_index('Device', inplace=True)
devices = list(df_time.index)
products = list(df_time.columns)
df_cap.set_index('Device', inplace=True)
capacity = df_cap['Monthly_Capacity'].to_dict()
capacity = {d: capacity[d] for d in devices if d in capacity}
df_profit.set_index('Product', inplace=True)
profit = df_profit['Unit_Profit'].to_dict()
profit = {p: profit[p] for p in products if p in profit}
m = gp.Model('MaxProfit')
x = m.addVars(products)
m.setObjective(gp.quicksum((profit[p] * x[p] for p in products)), gp.GRB.MAXIMIZE)
for d in devices:
    m.addConstr(gp.quicksum((df_time.loc[d, p] * x[p] for p in products)) <= capacity[d])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('\nProduction plan:')
    for p in products:
        qty = x[p].X
        if qty > 1e-06:
            print(f'  {p}: {qty:.2f} units')
else:
    print(f'Optimization ended with status {m.status}')