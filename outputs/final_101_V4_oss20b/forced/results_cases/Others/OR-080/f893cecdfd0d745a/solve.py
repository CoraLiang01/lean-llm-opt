import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv', sep=',')
trucks = df['truck_id'].tolist()
periods = [1, 2, 3, 4]
Q = dict(zip(df['truck_id'], df['Q']))
S = dict(zip(df['truck_id'], df['S']))
C = dict(zip(df['truck_id'], df['C']))
demand = {1: df.loc[0, 'd1'], 2: df.loc[0, 'd2'], 3: df.loc[0, 'd3'], 4: df.loc[0, 'd4']}
m = gp.Model('TruckScheduling')
a = m.addVars(periods, trucks, vtype=gp.GRB.BINARY)
s = m.addVars(periods, trucks, vtype=gp.GRB.BINARY)
d = m.addVars([1, 2, 3], trucks, vtype=gp.GRB.BINARY)
w = m.addVars(periods, trucks, vtype=gp.GRB.CONTINUOUS)
m.setObjective(gp.quicksum((C[i] * w[t, i] + S[i] * s[t, i] for t in periods for i in trucks)), gp.GRB.MINIMIZE)
m.addConstrs((a[1, i] == s[1, i] for i in trucks))
m.addConstrs((a[t, i] == a[t - 1, i] + s[t, i] - d[t - 1, i] for t in [2, 3, 4] for i in trucks))
m.addConstrs((s[t, i] <= 1 - a[t - 1, i] for t in [2, 3, 4] for i in trucks))
m.addConstrs((s[4, i] == 0 for i in trucks))
m.addConstrs((d[t, i] <= a[t, i] for t in [1, 2, 3] for i in trucks))
m.addConstrs((a[t + 1, i] <= 1 - d[t, i] for t in [1, 2, 3] for i in trucks))
m.addConstrs((a[t + 2, i] <= 1 - d[t, i] for t in [1, 2] for i in trucks))
m.addConstrs((a[3, i] == a[4, i] for i in trucks))
m.addConstrs((a[2, i] <= a[3, i] for i in trucks))
m.addConstrs((w[t, i] <= Q[i] * a[t, i] for t in periods for i in trucks))
m.addConstrs((w[t, i] - w[t - 1, i] <= 300 for t in [2, 3, 4] for i in trucks))
m.addConstrs((w[t - 1, i] - w[t, i] <= 300 for t in [2, 3, 4] for i in trucks))
m.addConstrs((gp.quicksum((w[t, i] for i in trucks)) >= demand[t] for t in periods))
m.addConstrs((gp.quicksum((w[t, i] for i in trucks)) <= 0.9 * gp.quicksum((Q[i] * a[t, i] for i in trucks)) for t in periods))
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal objective value: {m.objVal:.2f}')
    print('\n--- Truck Activation and Load Schedule ---')
    for t in periods:
        print(f'\nPeriod {t}:')
        for i in trucks:
            active = a[t, i].X
            load = w[t, i].X
            if active > 0.5:
                print(f'  Truck {i}: ACTIVE, Load = {load:.1f} kg')
            else:
                print(f'  Truck {i}: INACTIVE, Load = {load:.1f} kg')
else:
    print(f'No optimal solution found. Status: {m.status}')