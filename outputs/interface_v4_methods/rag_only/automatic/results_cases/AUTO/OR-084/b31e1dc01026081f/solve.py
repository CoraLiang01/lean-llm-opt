import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_problem():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
    df = pd.read_csv(path, sep=',')
    task_ids = [str(i) for i in range(1, 41)]
    cpu_ids = [1, 2, 3]
    cpu_freqs = {1: 1.33, 2: 2.0, 3: 2.66}
    bi_row = df[df['Process'].astype(str).str.casefold().str.strip() == 'bi']
    if bi_row.empty:
        raise ValueError("No row with Process == 'BI' found in the CSV.")
    bi_row = bi_row.iloc[0]
    bi_dict = {}
    for t in task_ids:
        val = bi_row[t]
        try:
            bi_dict[t] = float(val)
        except Exception:
            raise ValueError(f'Cannot convert BI value for task {t}: {val}')
    proc_time = {}
    for t in task_ids:
        proc_time[t] = {}
        for p in cpu_ids:
            proc_time[t][p] = bi_dict[t] / cpu_freqs[p]
    m = gp.Model('makespan_parallel_machine')
    assign = m.addVars(task_ids, cpu_ids, vtype=GRB.BINARY, lb=0, ub=1, name='')
    Cmax = m.addVar(vtype=GRB.CONTINUOUS, name='Cmax', lb=0)
    for t in task_ids:
        m.addConstr(gp.quicksum((assign[t, p] for p in cpu_ids)) == 1, name=f'assign_{t}')
    for p in cpu_ids:
        m.addConstr(gp.quicksum((proc_time[t][p] * assign[t, p] for t in task_ids)) <= Cmax, name=f'cpu_load_{p}')
    m.setObjective(Cmax, GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_problem()