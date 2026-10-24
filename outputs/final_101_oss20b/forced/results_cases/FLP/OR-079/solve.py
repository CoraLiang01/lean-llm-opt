CSVQA_DATA = {'query': 'ElectroTech Manufacturing is undergoing a critical strategic network restructuring. The primary objective is to minimize the total system cost by optimizing the supply chain backbone. This requires a dual decision: 1) Determine which subset of 15 potential factory sites (A1-A15) to construct (considering fixed costs from facility_costs.csv). 2) Plan the most economical shipment routes from these selected factories to satisfy the fixed demand of 8 distribution centers (B1-B8) (demand from demand_requirements.csv; variable costs from shipping_costs.csv).The core task is to find the optimal balance between fixed investment and variable logistics costs while ensuring all market demand is met.', 'relationships': [{'column_axis': {'id_column': 'Destination', 'table_id': 'file_2_view_0'}, 'matrix_table_id': 'file_1_view_0', 'row_axis': {'id_column': 'Facility', 'table_id': 'file_0_view_0'}, 'row_id_column': 'Origin', 'type': 'matrix'}], 'tables': [{'columns': ['Facility', 'FixedCost', 'Capacity'], 'file_name': 'facility_costs.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '30', 'Facility': 'A1', 'FixedCost': '0'}}, {'data_row_number': 1, 'values': {'Capacity': '10', 'Facility': 'A2', 'FixedCost': '175'}}, {'data_row_number': 2, 'values': {'Capacity': '20', 'Facility': 'A3', 'FixedCost': '300'}}, {'data_row_number': 3, 'values': {'Capacity': '30', 'Facility': 'A4', 'FixedCost': '375'}}, {'data_row_number': 4, 'values': {'Capacity': '40', 'Facility': 'A5', 'FixedCost': '500'}}, {'data_row_number': 5, 'values': {'Capacity': '20', 'Facility': 'A6', 'FixedCost': '200'}}, {'data_row_number': 6, 'values': {'Capacity': '25', 'Facility': 'A7', 'FixedCost': '260'}}, {'data_row_number': 7, 'values': {'Capacity': '30', 'Facility': 'A8', 'FixedCost': '220'}}, {'data_row_number': 8, 'values': {'Capacity': '35', 'Facility': 'A9', 'FixedCost': '320'}}, {'data_row_number': 9, 'values': {'Capacity': '20', 'Facility': 'A10', 'FixedCost': '280'}}, {'data_row_number': 10, 'values': {'Capacity': '40', 'Facility': 'A11', 'FixedCost': '350'}}, {'data_row_number': 11, 'values': {'Capacity': '25', 'Facility': 'A12', 'FixedCost': '420'}}, {'data_row_number': 12, 'values': {'Capacity': '30', 'Facility': 'A13', 'FixedCost': '470'}}, {'data_row_number': 13, 'values': {'Capacity': '50', 'Facility': 'A14', 'FixedCost': '520'}}, {'data_row_number': 14, 'values': {'Capacity': '45', 'Facility': 'A15', 'FixedCost': '560'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Origin', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'], 'file_name': 'shipping_costs.csv', 'records': [{'data_row_number': 0, 'values': {'B1': '8', 'B2': '4', 'B3': '3', 'B4': '6', 'B5': '7', 'B6': '5', 'B7': '9', 'B8': '8', 'Origin': 'A1'}}, {'data_row_number': 1, 'values': {'B1': '5', 'B2': '2', 'B3': '3', 'B4': '5', 'B5': '6', 'B6': '4', 'B7': '7', 'B8': '6', 'Origin': 'A2'}}, {'data_row_number': 2, 'values': {'B1': '4', 'B2': '3', 'B3': '4', 'B4': '6', 'B5': '5', 'B6': '5', 'B7': '6', 'B8': '7', 'Origin': 'A3'}}, {'data_row_number': 3, 'values': {'B1': '9', 'B2': '7', 'B3': '5', 'B4': '8', 'B5': '9', 'B6': '6', 'B7': '10', 'B8': '7', 'Origin': 'A4'}}, {'data_row_number': 4, 'values': {'B1': '10', 'B2': '4', 'B3': '2', 'B4': '6', 'B5': '8', 'B6': '5', 'B7': '7', 'B8': '3', 'Origin': 'A5'}}, {'data_row_number': 5, 'values': {'B1': '6', 'B2': '5', 'B3': '4', 'B4': '5', 'B5': '7', 'B6': '6', 'B7': '8', 'B8': '5', 'Origin': 'A6'}}, {'data_row_number': 6, 'values': {'B1': '7', 'B2': '6', 'B3': '5', 'B4': '4', 'B5': '6', 'B6': '7', 'B7': '9', 'B8': '6', 'Origin': 'A7'}}, {'data_row_number': 7, 'values': {'B1': '5', 'B2': '4', 'B3': '6', 'B4': '3', 'B5': '5', 'B6': '6', 'B7': '7', 'B8': '6', 'Origin': 'A8'}}, {'data_row_number': 8, 'values': {'B1': '8', 'B2': '7', 'B3': '6', 'B4': '7', 'B5': '9', 'B6': '8', 'B7': '10', 'B8': '7', 'Origin': 'A9'}}, {'data_row_number': 9, 'values': {'B1': '6', 'B2': '5', 'B3': '7', 'B4': '4', 'B5': '6', 'B6': '5', 'B7': '7', 'B8': '5', 'Origin': 'A10'}}, {'data_row_number': 10, 'values': {'B1': '9', 'B2': '6', 'B3': '4', 'B4': '6', 'B5': '8', 'B6': '7', 'B7': '9', 'B8': '6', 'Origin': 'A11'}}, {'data_row_number': 11, 'values': {'B1': '7', 'B2': '5', 'B3': '6', 'B4': '5', 'B5': '6', 'B6': '5', 'B7': '8', 'B8': '5', 'Origin': 'A12'}}, {'data_row_number': 12, 'values': {'B1': '8', 'B2': '6', 'B3': '5', 'B4': '6', 'B5': '7', 'B6': '6', 'B7': '8', 'B8': '7', 'Origin': 'A13'}}, {'data_row_number': 13, 'values': {'B1': '9', 'B2': '5', 'B3': '3', 'B4': '5', 'B5': '7', 'B6': '4', 'B7': '6', 'B8': '4', 'Origin': 'A14'}}, {'data_row_number': 14, 'values': {'B1': '10', 'B2': '6', 'B3': '4', 'B4': '5', 'B5': '8', 'B6': '5', 'B7': '7', 'B8': '5', 'Origin': 'A15'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Destination', 'Demand'], 'file_name': 'demand_requirements.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '30', 'Destination': 'B1'}}, {'data_row_number': 1, 'values': {'Demand': '25', 'Destination': 'B2'}}, {'data_row_number': 2, 'values': {'Demand': '20', 'Destination': 'B3'}}, {'data_row_number': 3, 'values': {'Demand': '35', 'Destination': 'B4'}}, {'data_row_number': 4, 'values': {'Demand': '25', 'Destination': 'B5'}}, {'data_row_number': 5, 'values': {'Demand': '30', 'Destination': 'B6'}}, {'data_row_number': 6, 'values': {'Demand': '25', 'Destination': 'B7'}}, {'data_row_number': 7, 'values': {'Demand': '30', 'Destination': 'B8'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB
tables = CSVQA_DATA['tables']

def get_table(table_id):
    for t in tables:
        if t['table_id'] == table_id:
            return t
    raise ValueError(f'Table {table_id} not found')
fac_table = get_table('file_0_view_0')
fac_records = fac_table['records']
I = []
f = {}
cap = {}
for rec in fac_records:
    vals = rec['values']
    fac_id = vals['Facility']
    I.append(fac_id)
    f[fac_id] = float(vals['FixedCost'])
    cap[fac_id] = float(vals['Capacity'])
ship_table = get_table('file_1_view_0')
ship_records = ship_table['records']
J = [col for col in ship_table['columns'] if col != 'Origin']
c = {}
for rec in ship_records:
    vals = rec['values']
    origin = vals['Origin']
    if origin not in I:
        raise ValueError(f'Origin {origin} not in facility list')
    c[origin] = {}
    for dest in J:
        if dest not in vals:
            raise ValueError(f'Missing cost for {origin}->{dest}')
        c[origin][dest] = float(vals[dest])
dem_table = get_table('file_2_view_0')
dem_records = dem_table['records']
d = {}
for rec in dem_records:
    vals = rec['values']
    dest = vals['Destination']
    if dest not in J:
        raise ValueError(f'Destination {dest} not in shipping matrix')
    d[dest] = float(vals['Demand'])
for i in I:
    if i not in f or i not in cap:
        raise ValueError(f'Missing parameters for facility {i}')
for j in J:
    if j not in d:
        raise ValueError(f'Missing demand for destination {j}')
for i in I:
    for j in J:
        if j not in c[i]:
            raise ValueError(f'Missing shipping cost for {i}->{j}')

def solve_problem():
    m = gurobipy.Model('FactoryLocationProblem')
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((f[i] * y[i] for i in I)) + gurobipy.quicksum((c[i][j] * x[i, j] for i in I for j in J))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in J:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in I)) == d[j], name=f'demand_{j}')
    for i in I:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in J)) <= cap[i] * y[i], name=f'capacity_{i}')
    m.optimize()
    return m
m = solve_problem()
