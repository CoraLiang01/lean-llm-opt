import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
costs_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv'
fixed_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv'
df_costs = pd.read_csv(costs_path)
df_fixed = pd.read_csv(fixed_path)
I = list(df_costs['Customer'])
J = list(df_fixed['Service Center'])
cost_dict = df_costs.set_index('Customer').to_dict(orient='index')
fixed_dict = dict(zip(df_fixed['Service Center'], df_fixed['Fixed Opening Cost']))
m = gp.Model('UFLP')
y = m.addVars(J, vtype=gp.GRB.BINARY)
x = m.addVars(I, J, vtype=gp.GRB.BINARY)
m.setObjective(gp.quicksum((fixed_dict[j] * y[j] for j in J)) + gp.quicksum((cost_dict[i][j] * x[i, j] for i in I for j in J)), gp.GRB.MINIMIZE)
m.addConstrs(((x[i, :].sum() == 1, f'assign_{i}') for i in I))
m.addConstrs(((x[i, j] <= y[j], f'link_{i}_{j}') for i in I for j in J))
m.addConstrs(((x[:, j].sum() <= 4, f'cap_{j}') for j in J))
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\nOpened service centres:')
    for j in J:
        if y[j].X > 0.5:
            print(f'  {j} (fixed cost {fixed_dict[j]:.2f})')
    print('\nCustomer assignments:')
    for i in I:
        for j in J:
            if x[i, j].X > 0.5:
                print(f'  {i} -> {j} (service cost {cost_dict[i][j]:.2f})')
else:
    print(f'Solver status: {m.status}')