import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# Path to the prepared inputs JSON file
DATA_PATH = "work/Variant1/prepared_inputs.json"

def solve_problem():
    # Load the prepared tables
    data = pd.read_json(DATA_PATH, typ='series')
    tables = {t['name']: t['rows'] for t in data['tables']}

    # Extract the monthly lot‑sizing data
    rows = tables['monthly_lot_sizing']
    months = [row['Month'] for row in rows]

    # Convert string values to numeric types
    d_t = {}
    c_t = {}
    s_t = {}
    h_t = {}
    p_t = {}
    for row in rows:
        month = row['Month']
        d_t[month] = int(row['Demand'])
        c_t[month] = float(row['ProductionCost'])
        s_t[month] = float(row['SetupCost'])
        h_t[month] = float(row['HoldingCost'])
        p_t[month] = int(row['ProductionCapacity'])

    # Create the Gurobi model
    m = gp.Model('LotSizing')
    m.setParam('MIPGap', 1e-6)

    # Decision variables
    x = m.addVars(months, vtype=GRB.CONTINUOUS, lb=0, name='x')
    I = m.addVars(months, vtype=GRB.CONTINUOUS, lb=0, name='I')
    y = m.addVars(months, vtype=GRB.BINARY, name='y')

    # Objective: minimize production, setup, and holding costs
    obj = gp.quicksum(c_t[mth] * x[mth] + s_t[mth] * y[mth] + h_t[mth] * I[mth] for mth in months)
    m.setObjective(obj, GRB.MINIMIZE)

    # Inventory balance constraints
    for idx, mth in enumerate(months):
        if idx == 0:
            # Initial inventory is zero
            m.addConstr(x[mth] - I[mth] == d_t[mth], name=f'bal_{mth}')
        else:
            prev_mth = months[idx - 1]
            m.addConstr(I[prev_mth] + x[mth] == d_t[mth] + I[mth], name=f'bal_{mth}')

    # Production capacity / setup linking
    for mth in months:
        m.addConstr(x[mth] <= p_t[mth] * y[mth], name=f'cap_{mth}')

    # Final inventory must be zero
    m.addConstr(I[months[-1]] == 0, name='final_inv')

    # (Inventory non‑negativity is already enforced by lb=0 on I)

    # Optimize
    m.optimize()

    return m

# Expose the model for inspection
m = solve_problem()