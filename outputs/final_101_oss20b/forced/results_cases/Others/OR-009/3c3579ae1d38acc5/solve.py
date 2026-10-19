import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
demand_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv')
supply_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv')
cost_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv')
customers = demand_df['customer'].tolist()
plants = supply_df['Unnamed: 0'].tolist()
demand = dict(zip(customers, demand_df['demand']))
supply_capacity = dict(zip(plants, supply_df['supply_capacity']))
cost_df_indexed = cost_df.set_index('Unnamed: 0')
cost = {(i, j): cost_df_indexed.loc[i, j] for i in plants for j in customers}
m = gp.Model('Transportation')
x = m.addVars(plants, customers)
m.setObjective(gp.quicksum((cost[i, j] * x[i, j] for i in plants for j in customers)), gp.GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in plants)) == demand[j] for j in customers))
m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i] for i in plants))
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('Shipment plan (units):')
    for i in plants:
        for j in customers:
            val = x[i, j].X
            if val > 1e-06:
                print(f'  {i} -> {j}: {val:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')