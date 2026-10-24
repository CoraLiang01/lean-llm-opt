import pandas as pd
import gurobipy
from gurobipy import GRB

# Path to the prepared inputs JSON
DATA_PATH = "work/Variant8/prepared_inputs.json"

def solve_problem():
    # Load the prepared data
    data = pd.read_json(DATA_PATH, typ='series')
    tables = {t['name']: t['rows'] for t in data['tables']}

    # Extract neighborhood demand
    demand_rows = tables['neighborhood_demand']
    demand = {row['Neighborhood']: float(row['Demand']) for row in demand_rows}
    neighborhood_ids = list(demand.keys())

    # Extract clinic distances
    distance_rows = tables['clinic_distance']
    distances = {}
    for row in distance_rows:
        clinic = row['Clinic']
        distances[clinic] = {}
        for col, val in row.items():
            if col == 'Clinic':
                continue
            distances[clinic][col] = float(val)
    clinic_ids = list(distances.keys())

    # Extract planning parameters
    param_rows = tables['planning_parameters']
    num_clinics_to_open = None
    for row in param_rows:
        if row['Parameter'] == 'NumberOfClinicsToOpen':
            num_clinics_to_open = int(row['Value'])
            break
    if num_clinics_to_open is None:
        raise ValueError('NumberOfClinicsToOpen parameter not found')

    # Basic data validation
    if num_clinics_to_open > len(clinic_ids):
        raise ValueError('Number of clinics to open exceeds available clinics')
    for clinic in clinic_ids:
        for neigh in neighborhood_ids:
            if neigh not in distances[clinic]:
                raise ValueError(f'Missing distance from clinic {clinic} to neighborhood {neigh}')
    for neigh in neighborhood_ids:
        if neigh not in demand:
            raise ValueError(f'Missing demand for neighborhood {neigh}')

    # Build the model
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)

    # Decision variables
    y = model.addVars(clinic_ids, vtype=GRB.BINARY, name='y')
    x = model.addVars([(i, j) for i in clinic_ids for j in neighborhood_ids],
                      vtype=GRB.BINARY, name='x')

    # Objective: minimize demand-weighted travel distance
    obj = gurobipy.quicksum(demand[j] * distances[i][j] * x[i, j]
                            for i in clinic_ids for j in neighborhood_ids)
    model.setObjective(obj, GRB.MINIMIZE)

    # Each neighborhood assigned to exactly one clinic
    for j in neighborhood_ids:
        model.addConstr(gurobipy.quicksum(x[i, j] for i in clinic_ids) == 1,
                        name=f'assign_{j}')

    # Linking constraint: assignment only if clinic is opened
    for i in clinic_ids:
        for j in neighborhood_ids:
            model.addConstr(x[i, j] <= y[i], name=f'link_{i}_{j}')

    # Exactly the required number of clinics are opened
    model.addConstr(gurobipy.quicksum(y[i] for i in clinic_ids) == num_clinics_to_open,
                    name='open_count')

    # Optimize
    model.optimize()

    return model

# Expose the model for inspection
m = solve_problem()