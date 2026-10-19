import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
school_capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv'
neighborhoods_population_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv'
distance_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv'
school_df = pd.read_csv(school_capacity_path, sep=',')
pop_df = pd.read_csv(neighborhoods_population_path, sep=',')
dist_df = pd.read_csv(distance_path, sep=',')
capacity = dict(zip(school_df['School'].astype(str), school_df['Capacity'].astype(int)))
pop_white = dict(zip(pop_df['Neighborhood'].astype(str), pop_df['Population_White'].astype(int)))
pop_nonwhite = dict(zip(pop_df['Neighborhood'].astype(str), pop_df['Population_NonWhite'].astype(int)))
dist_df = dist_df.set_index('School')
dist_df.columns = dist_df.columns.astype(str)
dist = {s: dict(zip(dist_df.columns, dist_df.loc[s].values)) for s in dist_df.index.astype(str)}
schools = list(capacity.keys())
neighborhoods = list(pop_white.keys())
races = ['White', 'NonWhite']
m = gp.Model('SchoolBusing')
x = m.addVars(schools, neighborhoods, races, vtype=gp.GRB.CONTINUOUS)
obj = gp.quicksum((dist[s][n] * x[s, n, r] for s in schools for n in neighborhoods for r in races))
m.setObjective(obj, gp.GRB.MINIMIZE)
for s in schools:
    m.addConstr(gp.quicksum((x[s, n, r] for n in neighborhoods for r in races)) <= capacity[s])
for n in neighborhoods:
    m.addConstr(gp.quicksum((x[s, n, 'White'] for s in schools)) == pop_white[n])
    m.addConstr(gp.quicksum((x[s, n, 'NonWhite'] for s in schools)) == pop_nonwhite[n])
p_white = 0.6
delta = 0.1
p_min = p_white - delta
p_max = p_white + delta
for s in schools:
    total = gp.quicksum((x[s, n, r] for n in neighborhoods for r in races))
    white = gp.quicksum((x[s, n, 'White'] for n in neighborhoods))
    m.addConstr(white >= p_min * total)
    m.addConstr(white <= p_max * total)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.4f} miles\n')
    for s in schools:
        total_assigned = sum((x[s, n, r].X for n in neighborhoods for r in races))
        white_assigned = sum((x[s, n, 'White'].X for n in neighborhoods))
        nonwhite_assigned = sum((x[s, n, 'NonWhite'].X for n in neighborhoods))
        white_ratio = white_assigned / total_assigned if total_assigned > 0 else 0
        print(f'School {s}:')
        print(f'  Total assigned: {total_assigned:.0f} students (Capacity: {capacity[s]})')
        print(f'  White students: {white_assigned:.0f} ({white_ratio * 100:.1f}%)')
        print(f'  Non‑White students: {nonwhite_assigned:.0f} ({(1 - white_ratio) * 100:.1f}%)')
        print('  Assignments by neighborhood and race:')
        for n in neighborhoods:
            for r in races:
                val = x[s, n, r].X
                if val > 1e-06:
                    print(f'    {n} {r}: {val:.0f}')
        print()
else:
    print(f'No optimal solution found. Status: {m.status}')