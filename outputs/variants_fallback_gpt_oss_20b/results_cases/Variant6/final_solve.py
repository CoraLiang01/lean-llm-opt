import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# Path to the prepared JSON data
DATA_PATH = "work/Variant6/prepared_inputs.json"

# Load the prepared tables
data = pd.read_json(DATA_PATH, typ='series')
tables = {t['name']: t['rows'] for t in data['tables']}

# Extract option catalog and resource limits
option_rows = tables['option_catalog']
resource_rows = tables['resource_limits']

# Build sets and parameter dictionaries
families = set()
options_per_family = {}
v = {}
w = {}
b = {}

for rec in option_rows:
    f = rec['Family']
    o = rec['Option']
    families.add(f)
    options_per_family.setdefault(f, set()).add(o)
    v[f, o] = int(rec['Value'])
    w[f, o] = int(rec['Weight'])
    b[f, o] = int(rec['BudgetUse'])

# Resource limits
Wmax = None
Bmax = None
for rec in resource_rows:
    res = rec['Resource']
    limit = int(rec['Limit'])
    if res == 'Weight':
        Wmax = limit
    elif res == 'BudgetUse':
        Bmax = limit

if Wmax is None or Bmax is None:
    raise ValueError('Missing resource limits')

# Create the model
m = gp.Model('multi_choice_knapsack')
m.setParam('MIPGap', 1e-06)

# Decision variables
x = m.addVars(
    [(f, o) for f in families for o in options_per_family[f]],
    vtype=GRB.BINARY,
    name='x'
)

# Objective: maximize total value
m.setObjective(
    gp.quicksum(v[f, o] * x[f, o] for f in families for o in options_per_family[f]),
    GRB.MAXIMIZE
)

# Constraints
# One option per family
for f in families:
    m.addConstr(
        gp.quicksum(x[f, o] for o in options_per_family[f]) == 1,
        name=f'one_option_{f}'
    )

# Weight limit
m.addConstr(
    gp.quicksum(w[f, o] * x[f, o] for f in families for o in options_per_family[f]) <= Wmax,
    name='weight_limit'
)

# Budget limit
m.addConstr(
    gp.quicksum(b[f, o] * x[f, o] for f in families for o in options_per_family[f]) <= Bmax,
    name='budget_limit'
)

# Optimize
m.optimize()

# Expose the model for inspection
model = m

# Print results
if m.status == GRB.OPTIMAL:
    print(f"Optimal objective: {m.ObjVal}")
    for var in m.getVars():
        print(f"{var.VarName} = {var.X}")
else:
    print(f"Optimization ended with status {m.status}")