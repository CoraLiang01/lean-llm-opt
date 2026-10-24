import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# Path to the prepared JSON file containing the tables
DATA_PATH = "work/Variant14/prepared_inputs.json"

def solve_problem():
    # Load the prepared data
    data = pd.read_json(DATA_PATH, typ='series')
    tables = {t['name']: t['rows'] for t in data['tables']}

    facility_table = tables['facility_sites']
    service_table = tables['service_zones']

    # Build sets and parameters
    I = []          # depot identifiers
    c = {}          # opening costs
    coverage = {}   # zones covered by each depot

    for rec in facility_table:
        center = rec['Center']
        if center in I:
            raise ValueError(f'Duplicate depot identifier {center}')
        I.append(center)
        c[center] = float(rec['OpeningCost'])
        zones = [z.strip() for z in rec['CoveredDistricts'].split(';') if z.strip()]
        coverage[center] = set(zones)

    J = []          # zone identifiers
    for rec in service_table:
        zone = rec['Zone']
        if zone in J:
            raise ValueError(f'Duplicate zone identifier {zone}')
        J.append(zone)

    # Ensure every zone is covered by at least one depot
    for zone in J:
        if not any(zone in coverage[center] for center in I):
            raise ValueError(f'Zone {zone} is not covered by any depot')

    # Build the optimization model
    m = gp.Model()
    m.setParam('MIPGap', 1e-6)

    y = m.addVars(I, vtype=GRB.BINARY, name='')

    m.setObjective(gp.quicksum(c[i] * y[i] for i in I), GRB.MINIMIZE)

    for zone in J:
        m.addConstr(
            gp.quicksum(y[i] for i in I if zone in coverage[i]) >= 1,
            name=f'cover_{zone}'
        )

    m.optimize()

    if m.Status == GRB.OPTIMAL:
        print(f'Optimal objective value: {m.ObjVal:.6f}')
        for var in m.getVars():
            print(f'{var.VarName} = {var.X:.6f}')
    else:
        print(f'Solver status: {m.Status}')

    return m

# Execute the solver
m = solve_problem()