import pandas as pd
import gurobipy
from gurobipy import GRB

# Path to the prepared inputs JSON file
data_path = "work/Variant27/prepared_inputs.json"

# Load the JSON data
CSVQA_DATA = pd.read_json(data_path, typ='series').to_dict()

# Map table IDs to table data
tables_by_id = {table['id']: table for table in CSVQA_DATA['tables']}

# Skill rank mapping
skill_rank_map = {'Junior': 1, 'Intermediate': 2, 'Senior': 3, 'Expert': 4}

def build_model():
    # Extract workers (exclude those on leave)
    workers = []
    worker_skill = {}
    for rec in tables_by_id['f000']['rows']:
        if rec.get('on_leave') == '0':
            wid = rec['worker_id']
            workers.append(wid)
            worker_skill[wid] = skill_rank_map[rec['skill']]

    # Extract projects
    projects = []
    project_req = {}
    for rec in tables_by_id['f001']['rows']:
        pid = rec['project_id']
        projects.append(pid)
        project_req[pid] = skill_rank_map[rec['required_skill']]

    # Cost files
    cost_file_ids = ['f002', 'f003', 'f004']
    cost_records = {}
    for fid in cost_file_ids:
        cost_records[fid] = {rec['worker_id']: rec for rec in tables_by_id[fid]['rows']}

    # Build cost and feasibility dictionaries
    cost = {}
    delta = {}
    for w in workers:
        for p in projects:
            c_val = None
            for fid in cost_file_ids:
                rec = cost_records[fid].get(w)
                if rec is None:
                    continue
                val = rec.get(p, '')
                if val != '':
                    c_val = int(val)
                    break
            if c_val is not None:
                cost[w, p] = c_val
                delta[w, p] = 1
            else:
                delta[w, p] = 0

    # Create Gurobi model
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)

    # Decision variables
    x = {}
    for w in workers:
        for p in projects:
            if delta[w, p] == 1:
                var = m.addVar(vtype=GRB.BINARY, name=f'x_{w}_{p}')
                x[w, p] = var

    m.update()

    # Objective
    obj = gurobipy.quicksum(cost[w, p] * x[w, p] for (w, p) in x)
    m.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    # Each project assigned to exactly one worker
    for p in projects:
        m.addConstr(
            gurobipy.quicksum(x[w, p] for w in workers if (w, p) in x) == 1,
            name=f'proj_{p}_assigned'
        )

    # Each worker takes at most one project
    for w in workers:
        m.addConstr(
            gurobipy.quicksum(x[w, p] for p in projects if (w, p) in x) <= 1,
            name=f'worker_{w}_max1'
        )

    # Skill compatibility
    M = 4
    for (w, p), var in x.items():
        m.addConstr(
            worker_skill[w] >= project_req[p] - M * (1 - var),
            name=f'skill_{w}_{p}'
        )

    # Forbidden assignments (redundant but kept for clarity)
    for (w, p), var in x.items():
        m.addConstr(
            var <= delta[w, p],
            name=f'forbidden_{w}_{p}'
        )

    m.update()
    m.optimize()
    return m

# Build and solve the model
m = build_model()