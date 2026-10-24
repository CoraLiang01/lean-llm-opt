import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# ------------------------------------------------------------------
# Load prepared data from the JSON file
# ------------------------------------------------------------------
DATA_PATH = (
    ""
    "work/Variant16/prepared_inputs.json"
)

data = pd.read_json(DATA_PATH, typ='series')
tables = {t['name']: t['rows'] for t in data['tables']}

option_catalog = tables['option_catalog']
resource_limits = tables['resource_limits']

# ------------------------------------------------------------------
# Prepare data structures
# ------------------------------------------------------------------
families = sorted({row['Family'] for row in option_catalog})
options_of_family = {
    fam: [row['Option'] for row in option_catalog if row['Family'] == fam]
    for fam in families
}

# Dictionaries for value, weight, and budget use keyed by (family, option)
v = {}
w = {}
b = {}
for row in option_catalog:
    key = (row['Family'], row['Option'])
    v[key] = float(row['Value'])
    w[key] = float(row['Weight'])
    b[key] = float(row['BudgetUse'])

# Resource limits
limits = {row['Resource']: float(row['Limit']) for row in resource_limits}
Wmax = limits['Weight']
Bmax = limits['BudgetUse']

# ------------------------------------------------------------------
# Build the optimization model with Gurobi
# ------------------------------------------------------------------
m = gp.Model("multi_choice_knapsack")

# Decision variables: x_{family}_{option}
x = {}
for fam in families:
    for opt in options_of_family[fam]:
        var = m.addVar(vtype=GRB.BINARY, name=f"x_{fam}_{opt}")
        x[(fam, opt)] = var

m.update()

# Objective: maximize total value
m.setObjective(gp.quicksum(v[key] * x[key] for key in x), GRB.MAXIMIZE)

# Exactly one option per family
for fam in families:
    m.addConstr(
        gp.quicksum(x[(fam, opt)] for opt in options_of_family[fam]) == 1,
        name=f"OneOption_{fam}"
    )

# Weight capacity constraint
m.addConstr(
    gp.quicksum(w[key] * x[key] for key in x) <= Wmax,
    name="WeightCapacity"
)

# Budget capacity constraint
m.addConstr(
    gp.quicksum(b[key] * x[key] for key in x) <= Bmax,
    name="BudgetCapacity"
)

# ------------------------------------------------------------------
# Solve the model
# ------------------------------------------------------------------
m.optimize()

# The model object is exposed as `m` for external inspection