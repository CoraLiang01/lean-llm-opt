import pandas as pd
import gurobipy
from gurobipy import GRB

# Path to the prepared JSON input
DATA_PATH = "work/Variant25/prepared_inputs.json"

# Skill level mapping
SKILL_MAP = {'Junior': 1, 'Intermediate': 2, 'Senior': 3, 'Expert': 4}

def solve_problem():
    # Load data
    data = pd.read_json(DATA_PATH, typ='series')
    tables = data['tables']

    # Retrieve tables by ID
    worker_table = next(t for t in tables if t['id'] == 'f000')
    project_table = next(t for t in tables if t['id'] == 'f001')
    cost_table = next(t for t in tables if t['id'] == 'f002')

    # Build worker set, filtering out those on leave
    workers = []
    skill_lvl = {}
    for rec in worker_table['rows']:
        if rec['on_leave'] == '0':
            w_id = rec['worker_id']
            skill_str = rec['skill']
            if skill_str not in SKILL_MAP:
                raise ValueError(f"Unknown skill level '{skill_str}' for worker {w_id}")
            workers.append(w_id)
            skill_lvl[w_id] = SKILL_MAP[skill_str]

    # Build project set
    projects = []
    req_lvl = {}
    for rec in project_table['rows']:
        p_id = rec['project_id']
        skill_str = rec['required_skill']
        if skill_str not in SKILL_MAP:
            raise ValueError(f"Unknown required skill '{skill_str}' for project {p_id}")
        projects.append(p_id)
        req_lvl[p_id] = SKILL_MAP[skill_str]

    # Map cost rows by worker_id, merging duplicates
    cost_rows = {}
    for rec in cost_table['rows']:
        w_id = rec['worker_id']
        if w_id in cost_rows:
            existing = cost_rows[w_id]
            for col, val in rec.items():
                if col == 'worker_id':
                    continue
                if val != '' and existing.get(col, '') == '':
                    existing[col] = val
        else:
            cost_rows[w_id] = rec

    # Validate that all workers have cost rows
    for w in workers:
        if w not in cost_rows:
            raise ValueError(f'Cost row missing for worker {w}')

    # Validate that all projects have cost columns
    cost_columns = [col for col in cost_table['columns'] if col != 'worker_id']
    for p in projects:
        if p not in cost_columns:
            raise ValueError(f'Cost column missing for project {p}')

    # Build cost and allowed matrices
    cost = {}
    allowed = {}
    for w in workers:
        row = cost_rows[w]
        cost[w] = {}
        allowed[w] = {}
        for p in projects:
            val = row.get(p, '')
            if val == '':
                cost[w][p] = 0
                allowed[w][p] = 0
            else:
                try:
                    c_val = int(val)
                except ValueError:
                    raise ValueError(f"Invalid cost value '{val}' for worker {w}, project {p}")
                cost[w][p] = c_val
                allowed[w][p] = 1 if skill_lvl[w] >= req_lvl[p] else 0

    # Create Gurobi model
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)

    # Decision variables
    x = m.addVars([(w, p) for w in workers for p in projects],
                  vtype=GRB.BINARY, name='')

    # Objective
    m.setObjective(gurobipy.quicksum(cost[w][p] * x[w, p] for w in workers for p in projects),
                   GRB.MINIMIZE)

    # Constraints
    for p in projects:
        m.addConstr(gurobipy.quicksum(x[w, p] for w in workers) == 1,
                    name=f'proj_{p}')

    for w in workers:
        m.addConstr(gurobipy.quicksum(x[w, p] for p in projects) <= 1,
                    name=f'worker_{w}')

    for w in workers:
        for p in projects:
            if allowed[w][p] == 0:
                m.addConstr(x[w, p] == 0, name=f'forbid_{w}_{p}')

    # Optimize
    m.optimize()
    return m

# Expose the model for inspection
m = solve_problem()