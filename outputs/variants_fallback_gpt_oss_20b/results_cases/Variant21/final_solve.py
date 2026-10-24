import pandas as pd
import gurobipy
from gurobipy import GRB

# Path to the prepared JSON file (exact string from the data section)
DATA_PATH = "work/Variant21/prepared_inputs.json"

def solve_problem():
    # Load the prepared data
    data = pd.read_json(DATA_PATH, typ='series')
    tables = {t['name']: t['rows'] for t in data['tables']}

    # Retrieve the activity and parameter tables
    activities_table = tables.get('project_activities')
    if activities_table is None:
        raise KeyError("Missing 'project_activities' table in the prepared data.")
    params_table = tables.get('project_parameters')
    if params_table is None:
        raise KeyError("Missing 'project_parameters' table in the prepared data.")

    # Parse activity data
    ND = {}   # Normal duration
    CD = {}   # Crash duration
    CC = {}   # Crash cost per day
    Pred = {} # Predecessor lists

    for record in activities_table:
        act = record['Activity']
        ND[act] = int(record['NormalDuration'])
        CD[act] = int(record['CrashDuration'])
        CC[act] = int(record['CrashCostPerDay'])
        preds_str = record['Predecessors']
        if preds_str.strip() == '':
            preds = []
        else:
            preds = [p.strip() for p in preds_str.split(';') if p.strip()]
        Pred[act] = preds

    # Validate durations
    for act in ND:
        if ND[act] < CD[act]:
            raise ValueError(f"Activity {act} has NormalDuration < CrashDuration.")
        if ND[act] - CD[act] < 0:
            raise ValueError(f"Activity {act} has negative crash limit.")

    # Validate predecessor references
    for act in Pred:
        for pred in Pred[act]:
            if pred not in ND:
                raise ValueError(f"Predecessor {pred} of activity {act} not found in activities.")

    # Retrieve project deadline
    PD = None
    for record in params_table:
        if record['Parameter'] == 'ProjectDeadline':
            PD = int(record['Value'])
            break
    if PD is None:
        raise ValueError("ProjectDeadline parameter not found in project_parameters table.")

    # Build the optimization model
    m = gurobipy.Model('ProjectCrashing')
    m.setParam('MIPGap', 1e-06)

    I = list(ND.keys())

    # Decision variables
    s = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='s_')
    z = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='z_')  # lower bound 0 added here
    T = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='T')

    # Objective: minimize total crashing cost
    m.setObjective(gurobipy.quicksum(CC[i] * z[i] for i in I), GRB.MINIMIZE)

    # Precedence constraints
    for i in I:
        for j in Pred[i]:
            m.addConstr(s[i] >= s[j] + ND[j] - z[j], name=f'prec_{i}_{j}')

    # Finish time constraints
    for i in I:
        m.addConstr(s[i] + ND[i] - z[i] <= T, name=f'finish_{i}')

    # Crash-day bounds
    for i in I:
        m.addConstr(z[i] <= ND[i] - CD[i], name=f'crashbound_{i}')

    # Project deadline
    m.addConstr(T <= PD, name='deadline')

    # Optimize
    m.optimize()

    return m

# Expose the model for inspection
m = solve_problem()