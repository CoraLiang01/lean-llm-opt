CSVQA_DATA = {'query': 'A public health agency wants to choose a fixed number of clinic sites to serve neighborhood demand. Neighborhood demand is given in neighborhood_demand.csv, travel distances from candidate clinics to neighborhoods are given in clinic_distance.csv, and the number of clinics that must be opened is given in planning_parameters.csv.\n\nFormulate a p-median location model. For each candidate clinic i, define y_i as a binary variable equal to 1 if clinic i is opened. For each clinic i and neighborhood j, define x_ij as a binary variable equal to 1 if neighborhood j is assigned to clinic i. The objective is to minimize total demand-weighted travel distance. The model should assign every neighborhood to exactly one clinic, open exactly the required number of clinics, link assignments to opened clinics, and impose binary restrictions on all variables.', 'relationships': [{'column_axis': {'id_column': 'Neighborhood', 'table_id': 'file_0_view_0'}, 'matrix_table_id': 'file_1_view_0', 'row_axis': {'id_column': 'Clinic', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Clinic', 'type': 'matrix'}], 'tables': [{'columns': ['Neighborhood', 'Demand'], 'file_name': 'neighborhood_demand.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '30', 'Neighborhood': 'N1'}}, {'data_row_number': 1, 'values': {'Demand': '45', 'Neighborhood': 'N2'}}, {'data_row_number': 2, 'values': {'Demand': '25', 'Neighborhood': 'N3'}}, {'data_row_number': 3, 'values': {'Demand': '50', 'Neighborhood': 'N4'}}, {'data_row_number': 4, 'values': {'Demand': '40', 'Neighborhood': 'N5'}}, {'data_row_number': 5, 'values': {'Demand': '35', 'Neighborhood': 'N6'}}, {'data_row_number': 6, 'values': {'Demand': '55', 'Neighborhood': 'N7'}}, {'data_row_number': 7, 'values': {'Demand': '20', 'Neighborhood': 'N8'}}, {'data_row_number': 8, 'values': {'Demand': '60', 'Neighborhood': 'N9'}}, {'data_row_number': 9, 'values': {'Demand': '30', 'Neighborhood': 'N10'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Clinic', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10'], 'file_name': 'clinic_distance.csv', 'records': [{'data_row_number': 0, 'values': {'Clinic': 'K1', 'N1': '2', 'N10': '16', 'N2': '3', 'N3': '9', 'N4': '10', 'N5': '11', 'N6': '12', 'N7': '13', 'N8': '14', 'N9': '15'}}, {'data_row_number': 1, 'values': {'Clinic': 'K2', 'N1': '3', 'N10': '15', 'N2': '2', 'N3': '8', 'N4': '9', 'N5': '10', 'N6': '11', 'N7': '12', 'N8': '13', 'N9': '14'}}, {'data_row_number': 2, 'values': {'Clinic': 'K3', 'N1': '10', 'N10': '13', 'N2': '9', 'N3': '2', 'N4': '3', 'N5': '4', 'N6': '9', 'N7': '10', 'N8': '11', 'N9': '12'}}, {'data_row_number': 3, 'values': {'Clinic': 'K4', 'N1': '11', 'N10': '12', 'N2': '10', 'N3': '3', 'N4': '2', 'N5': '5', 'N6': '8', 'N7': '9', 'N8': '10', 'N9': '11'}}, {'data_row_number': 4, 'values': {'Clinic': 'K5', 'N1': '13', 'N10': '9', 'N2': '12', 'N3': '10', 'N4': '9', 'N5': '8', 'N6': '2', 'N7': '3', 'N8': '4', 'N9': '8'}}, {'data_row_number': 5, 'values': {'Clinic': 'K6', 'N1': '14', 'N10': '2', 'N2': '13', 'N3': '11', 'N4': '10', 'N5': '9', 'N6': '3', 'N7': '2', 'N8': '5', 'N9': '3'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Parameter', 'Value'], 'file_name': 'planning_parameters.csv', 'records': [{'data_row_number': 0, 'values': {'Parameter': 'NumberOfClinicsToOpen', 'Value': '3'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']

    def get_table(table_id):
        for t in tables:
            if t['table_id'] == table_id:
                return t
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    demand_table = get_table('file_0_view_0')
    demand = {}
    for rec in demand_table['records']:
        values = rec['values']
        j = values['Neighborhood']
        d = float(values['Demand'])
        demand[j] = d
    J = list(demand.keys())
    distance_table = get_table('file_1_view_0')
    distance = {}
    for rec in distance_table['records']:
        values = rec['values']
        i = values['Clinic']
        row = {}
        for col in distance_table['columns']:
            if col == 'Clinic':
                continue
            if col in values:
                row[col] = float(values[col])
        distance[i] = row
    I = list(distance.keys())
    param_table = get_table('file_2_view_0')
    P = None
    for rec in param_table['records']:
        values = rec['values']
        if values['Parameter'] == 'NumberOfClinicsToOpen':
            P = int(float(values['Value']))
            break
    if P is None:
        raise ValueError('NumberOfClinicsToOpen parameter not found')
    if not I:
        raise ValueError('No clinics found')
    if not J:
        raise ValueError('No neighborhoods found')
    if P < 0 or P > len(I):
        raise ValueError(f'Invalid number of clinics to open: {P}')
    for i in I:
        if i not in distance:
            raise ValueError(f'Missing distance row for clinic {i}')
        for j in J:
            if j not in distance[i]:
                raise ValueError(f'Missing distance from clinic {i} to neighborhood {j}')
    for j in J:
        if j not in demand:
            raise ValueError(f'Missing demand for neighborhood {j}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.BINARY, name='')
    obj = gurobipy.quicksum((demand[j] * distance[i][j] * x[i, j] for i in I for j in J))
    m.setObjective(obj, GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in I)) == 1 for j in J), name='')
    m.addConstrs((x[i, j] <= y[i] for i in I for j in J), name='')
    m.addConstr(gurobipy.quicksum((y[i] for i in I)) == P, name='open_count')
    m.optimize()
    return m
m = solve_problem()
