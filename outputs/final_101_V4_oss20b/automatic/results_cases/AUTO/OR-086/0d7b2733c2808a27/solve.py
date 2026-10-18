import gurobipy as gp
import pandas as pd
import re
path_30_1 = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv'
path_30_2 = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv'
df1 = pd.read_csv(path_30_1, sep=',')
df2 = pd.read_csv(path_30_2, sep=',')
G = list(df1['Grade'])
B = list(df2['Brand'])
Supply = dict(zip(df1['Grade'], df1['Daily Supply (kg)']))
Cost = dict(zip(df1['Grade'], df1['Cost (CNY/kg)']))
Price = dict(zip(df2['Brand'], df2['Selling Price (CNY/kg)']))
lower_bounds = {}
upper_bounds = {}
pattern = re.compile('([A-Za-z]+)\\s+(less than|more than)\\s+(\\d+)%', re.IGNORECASE)
for _, row in df2.iterrows():
    brand = row['Brand']
    req_text = str(row['Blending Requirements']).strip()
    for match in pattern.findall(req_text):
        grade, op, pct = match
        pct_val = int(pct) / 100.0
        key = (brand, grade)
        if op.lower() == 'less than':
            upper_bounds[key] = pct_val
        else:
            lower_bounds[key] = pct_val
m = gp.Model('WineBlending')
x = m.addVars(G, B)
for g in G:
    m.addConstr(gp.quicksum((x[g, b] for b in B)) <= Supply[g])
m.addConstr(gp.quicksum((x[g, 'Red'] for g in G)) >= 2000)
for b in B:
    P_b = gp.quicksum((x[g, b] for g in G))
    for g in G:
        key = (b, g)
        if key in lower_bounds:
            m.addConstr(x[g, b] >= lower_bounds[key] * P_b)
        if key in upper_bounds:
            m.addConstr(x[g, b] <= upper_bounds[key] * P_b)
m.setObjective(gp.quicksum(((Price[b] - Cost[g]) * x[g, b] for g in G for b in B)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal net profit: {m.objVal:.2f} CNY')
    print('\nProduction plan (kg):')
    for b in B:
        total_b = sum((x[g, b].X for g in G))
        if total_b > 1e-06:
            print(f'  Brand {b}: {total_b:.2f} kg')
            for g in G:
                qty = x[g, b].X
                if qty > 1e-06:
                    print(f'    Grade {g}: {qty:.2f} kg')
else:
    print(f'Optimization did not find an optimal solution. Status: {m.status}')