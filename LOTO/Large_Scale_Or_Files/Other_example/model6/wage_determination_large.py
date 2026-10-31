#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic wage determination solver for mutual-work datasets of arbitrary size.
- Input: CSV with first column = Owner names, remaining columns = worker names.
- Constraint: For each i (owner), sum_j D[i,j] * w_j = 10 * w_i.
- Scale: Fix the first worker's wage to 60 yuan/day.
"""
import sys
import os
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def main(csv_path, script_dir=None):
    df = pd.read_csv(csv_path)
    owner_col = df.columns[0]
    worker_names = list(df.columns[1:])
    N = len(worker_names)
    assert df.shape[0] == N, "Rows (owners) must equal number of workers/columns."
    
    # Extract D (NxN)
    D = df[worker_names].to_numpy(dtype=float)
    
    # Basic sanity check: each column should sum to 10
    col_sums = D.sum(axis=0)
    if not np.allclose(col_sums, 10.0):
        raise ValueError("Each column (worker total days) should sum to 10.")
    
    # Build model
    m = gp.Model("WageDetermination_Large")
    m.Params.OutputFlag = 1  # set to 0 for quiet
    
    # Variables: wages >= 0
    w = m.addVars(N, lb=0.0, vtype=GRB.CONTINUOUS, name="wage")
    
    # Fix the first worker's wage as the scale
    m.addConstr(w[0] == 60.0, name=f"Fix_{worker_names[0]}")
    
    # Balance constraints: D w = 10 w
    for i in range(N):
        m.addConstr(gp.quicksum(D[i, j] * w[j] for j in range(N)) == 10.0 * w[i],
                    name=f"Balance[{i}]")
    
    # Feasibility objective
    m.setObjective(0.0, GRB.MINIMIZE)
    m.optimize()
    
    # Write LP file
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(csv_path))
    m.write(os.path.join(script_dir, "model.lp"))
    
    if m.status == GRB.OPTIMAL:
        print("\nOptimal Daily Wages (yuan/day):")
        for i, name in enumerate(worker_names):
            print(f"{name}: {w[i].X:.6f}")
        
        # Verification
        W = np.array([w[i].X for i in range(N)])
        income = 10.0 * W
        expenditure = D @ W
        max_imbalance = float(np.max(np.abs(income - expenditure)))
        print(f"\nMax absolute imbalance: {max_imbalance:.6e} (should be ~0)")
    else:
        print("No feasible solution. Solver status:", m.status)

if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(script_dir, "work_days.csv")
    csv_path = sys.argv[1] if len(sys.argv) > 1 else default_csv
    main(csv_path, script_dir=script_dir)
