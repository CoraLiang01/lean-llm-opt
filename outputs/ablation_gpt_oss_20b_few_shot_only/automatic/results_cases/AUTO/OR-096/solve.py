import gurobipy
from gurobipy import GRB
import pandas as pd

def solve_problem():
    school_capacity_path = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv'
    neighborhoods_population_path = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv'
    distance_path = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv'
    df_school = pd.read_csv(school_capacity_path, dtype=str, keep_default_na=False)
    df_school['School'] = df_school['School'].str.strip()
    df_school['Capacity'] = df_school['Capacity'].astype(float)
    capacity = dict(zip(df_school['School'], df_school['Capacity']))
    df_neigh = pd.read_csv(neighborhoods_population_path, dtype=str, keep_default_na=False)
    df_neigh['Neighborhood'] = df_neigh['Neighborhood'].str.strip()
    df_neigh['Population_White'] = df_neigh['Population_White'].astype(float)
    df_neigh['Population_NonWhite'] = df_neigh['Population_NonWhite'].astype(float)
    pop_white = dict(zip(df_neigh['Neighborhood'], df_neigh['Population_White']))
    pop_nonwhite = dict(zip(df_neigh['Neighborhood'], df_neigh['Population_NonWhite']))
    df_dist = pd.read_csv(distance_path, dtype=str, keep_default_na=False)
    df_dist.columns = df_dist.columns.str.strip()
    df_dist['School'] = df_dist['School'].str.strip()
    for col in df_dist.columns:
        if col != 'School':
            df_dist[col] = df_dist[col].astype(float)
    distance = {}
    for (_, row) in df_dist.iterrows():
        s = row['School']
        for n in pop_white.keys():
            distance[s, n] = row[n]
    schools = list(capacity.keys())
    neighborhoods = list(pop_white.keys())
    for s in schools:
        if s not in capacity:
            raise ValueError(f'Missing capacity for school {s}')
    for n in neighborhoods:
        if n not in pop_white or n not in pop_nonwhite:
            raise ValueError(f'Missing population data for neighborhood {n}')
    for s in schools:
        for n in neighborhoods:
            if (s, n) not in distance:
                raise ValueError(f'Missing distance for school {s} and neighborhood {n}')
    r = 0.6
    tau = 0.1
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    xW = m.addVars([(s, n) for s in schools for n in neighborhoods], lb=0, vtype=GRB.CONTINUOUS, name='')
    xNW = m.addVars([(s, n) for s in schools for n in neighborhoods], lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = gurobipy.quicksum((distance[s, n] * (xW[s, n] + xNW[s, n]) for s in schools for n in neighborhoods))
    m.setObjective(obj, GRB.MINIMIZE)
    for s in schools:
        m.addConstr(gurobipy.quicksum((xW[s, n] + xNW[s, n] for n in neighborhoods)) <= capacity[s], name=f'Cap_{s}')
    for n in neighborhoods:
        m.addConstr(gurobipy.quicksum((xW[s, n] for s in schools)) == pop_white[n], name=f'PopW_{n}')
        m.addConstr(gurobipy.quicksum((xNW[s, n] for s in schools)) == pop_nonwhite[n], name=f'PopNW_{n}')
    for s in schools:
        sum_xW = gurobipy.quicksum((xW[s, n] for n in neighborhoods))
        sum_total = gurobipy.quicksum((xW[s, n] + xNW[s, n] for n in neighborhoods))
        m.addConstr(sum_xW >= (r - tau) * sum_total, name=f'RaceLow_{s}')
        m.addConstr(sum_xW <= (r + tau) * sum_total, name=f'RaceHigh_{s}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'Optimal objective value: {m.ObjVal:.4f}')
        for var in m.getVars():
            print(f'{var.VarName} = {var.X:.4f}')
    else:
        print(f'Optimization ended with status {m.Status}')
    return m
m = solve_problem()