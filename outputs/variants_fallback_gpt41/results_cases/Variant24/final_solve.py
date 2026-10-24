LEGACY_OBSERVATION = '[{"source":"benchmark_dataset/Variant24/inputs/neighborhood_demand.csv","values":{"Neighborhood":"N1","Demand":"28"}},{"source":"benchmark_dataset/Variant24/inputs/neighborhood_demand.csv","values":{"Neighborhood":"N2","Demand":"32"}},{"source":"benchmark_dataset/Variant24/inputs/neighborhood_demand.csv","values":{"Neighborhood":"N3","Demand":"44"}},{"source":"benchmark_dataset/Variant24/inputs/neighborhood_demand.csv","values":{"Neighborhood":"N4","Demand":"36"}},{"source":"benchmark_dataset/Variant24/inputs/neighborhood_demand.csv","values":{"Neighborhood":"N5","Demand":"52"}},{"source":"benchmark_dataset/Variant24/inputs/neighborhood_demand.csv","values":{"Neighborhood":"N6","Demand":"41"}},{"source":"benchmark_dataset/Variant24/inputs/neighborhood_demand.csv","values":{"Neighborhood":"N7","Demand":"25"}},{"source":"benchmark_dataset/Variant24/inputs/neighborhood_demand.csv","values":{"Neighborhood":"N8","Demand":"48"}},{"source":"benchmark_dataset/Variant24/inputs/neighborhood_demand.csv","values":{"Neighborhood":"N9","Demand":"55"}},{"source":"benchmark_dataset/Variant24/inputs/neighborhood_demand.csv","values":{"Neighborhood":"N10","Demand":"30"}},{"source":"benchmark_dataset/Variant24/inputs/facility_distance.csv","values":{"Facility":"F1","N1":"2","N2":"3","N3":"5","N4":"9","N5":"10","N6":"11","N7":"13","N8":"14","N9":"15","N10":"12"}},{"source":"benchmark_dataset/Variant24/inputs/facility_distance.csv","values":{"Facility":"F2","N1":"4","N2":"2","N3":"3","N4":"8","N5":"9","N6":"10","N7":"12","N8":"13","N9":"14","N10":"11"}},{"source":"benchmark_dataset/Variant24/inputs/facility_distance.csv","values":{"Facility":"F3","N1":"9","N2":"8","N3":"4","N4":"2","N5":"3","N6":"5","N7":"9","N8":"10","N9":"11","N10":"7"}},{"source":"benchmark_dataset/Variant24/inputs/facility_distance.csv","values":{"Facility":"F4","N1":"10","N2":"9","N3":"6","N4":"3","N5":"2","N6":"3","N7":"8","N8":"9","N9":"10","N10":"6"}},{"source":"benchmark_dataset/Variant24/inputs/facility_distance.csv","values":{"Facility":"F5","N1":"13","N2":"12","N3":"10","N4":"8","N5":"7","N6":"6","N7":"2","N8":"3","N9":"4","N10":"5"}},{"source":"benchmark_dataset/Variant24/inputs/facility_distance.csv","values":{"Facility":"F6","N1":"12","N2":"11","N3":"8","N4":"7","N5":"6","N6":"5","N7":"5","N8":"4","N9":"3","N10":"2"}},{"source":"benchmark_dataset/Variant24/inputs/planning_parameters.csv","values":{"Parameter":"NumberOfFacilitiesToOpen","Value":"2"}}]'
LEGACY_RECORDS = [{'source': 'benchmark_dataset/Variant24/inputs/neighborhood_demand.csv', 'values': {'Neighborhood': 'N1', 'Demand': '28'}}, {'source': 'benchmark_dataset/Variant24/inputs/neighborhood_demand.csv', 'values': {'Neighborhood': 'N2', 'Demand': '32'}}, {'source': 'benchmark_dataset/Variant24/inputs/neighborhood_demand.csv', 'values': {'Neighborhood': 'N3', 'Demand': '44'}}, {'source': 'benchmark_dataset/Variant24/inputs/neighborhood_demand.csv', 'values': {'Neighborhood': 'N4', 'Demand': '36'}}, {'source': 'benchmark_dataset/Variant24/inputs/neighborhood_demand.csv', 'values': {'Neighborhood': 'N5', 'Demand': '52'}}, {'source': 'benchmark_dataset/Variant24/inputs/neighborhood_demand.csv', 'values': {'Neighborhood': 'N6', 'Demand': '41'}}, {'source': 'benchmark_dataset/Variant24/inputs/neighborhood_demand.csv', 'values': {'Neighborhood': 'N7', 'Demand': '25'}}, {'source': 'benchmark_dataset/Variant24/inputs/neighborhood_demand.csv', 'values': {'Neighborhood': 'N8', 'Demand': '48'}}, {'source': 'benchmark_dataset/Variant24/inputs/neighborhood_demand.csv', 'values': {'Neighborhood': 'N9', 'Demand': '55'}}, {'source': 'benchmark_dataset/Variant24/inputs/neighborhood_demand.csv', 'values': {'Neighborhood': 'N10', 'Demand': '30'}}, {'source': 'benchmark_dataset/Variant24/inputs/facility_distance.csv', 'values': {'Facility': 'F1', 'N1': '2', 'N2': '3', 'N3': '5', 'N4': '9', 'N5': '10', 'N6': '11', 'N7': '13', 'N8': '14', 'N9': '15', 'N10': '12'}}, {'source': 'benchmark_dataset/Variant24/inputs/facility_distance.csv', 'values': {'Facility': 'F2', 'N1': '4', 'N2': '2', 'N3': '3', 'N4': '8', 'N5': '9', 'N6': '10', 'N7': '12', 'N8': '13', 'N9': '14', 'N10': '11'}}, {'source': 'benchmark_dataset/Variant24/inputs/facility_distance.csv', 'values': {'Facility': 'F3', 'N1': '9', 'N2': '8', 'N3': '4', 'N4': '2', 'N5': '3', 'N6': '5', 'N7': '9', 'N8': '10', 'N9': '11', 'N10': '7'}}, {'source': 'benchmark_dataset/Variant24/inputs/facility_distance.csv', 'values': {'Facility': 'F4', 'N1': '10', 'N2': '9', 'N3': '6', 'N4': '3', 'N5': '2', 'N6': '3', 'N7': '8', 'N8': '9', 'N9': '10', 'N10': '6'}}, {'source': 'benchmark_dataset/Variant24/inputs/facility_distance.csv', 'values': {'Facility': 'F5', 'N1': '13', 'N2': '12', 'N3': '10', 'N4': '8', 'N5': '7', 'N6': '6', 'N7': '2', 'N8': '3', 'N9': '4', 'N10': '5'}}, {'source': 'benchmark_dataset/Variant24/inputs/facility_distance.csv', 'values': {'Facility': 'F6', 'N1': '12', 'N2': '11', 'N3': '8', 'N4': '7', 'N5': '6', 'N6': '5', 'N7': '5', 'N8': '4', 'N9': '3', 'N10': '2'}}, {'source': 'benchmark_dataset/Variant24/inputs/planning_parameters.csv', 'values': {'Parameter': 'NumberOfFacilitiesToOpen', 'Value': '2'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
neighborhood_demand = {}
for rec in records:
    if rec['source'].endswith('neighborhood_demand.csv'):
        n = rec['values']['Neighborhood']
        d = int(rec['values']['Demand'])
        if n in neighborhood_demand:
            raise ValueError(f'Duplicate demand entry for {n}')
        neighborhood_demand[n] = d
facility_distance = {}
facilities = []
for rec in records:
    if rec['source'].endswith('facility_distance.csv'):
        i = rec['values']['Facility']
        facilities.append(i)
        facility_distance[i] = {}
        for j in rec['values']:
            if j != 'Facility':
                facility_distance[i][j] = int(rec['values'][j])
p = None
for rec in records:
    if rec['source'].endswith('planning_parameters.csv'):
        if rec['values'].get('Parameter') == 'NumberOfFacilitiesToOpen':
            p = int(rec['values']['Value'])
if p is None:
    raise ValueError('Number of facilities to open not found.')
neighborhoods = sorted(neighborhood_demand.keys())
facilities = sorted(facilities)
for i in facilities:
    if set(facility_distance[i].keys()) != set(neighborhoods):
        raise ValueError(f'Distance data for facility {i} does not cover all neighborhoods.')
m = gp.Model('p_median')
x_vars = m.addVars(facilities, neighborhoods, vtype=GRB.BINARY, name='')
y_vars = m.addVars(facilities, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((neighborhood_demand[j] * facility_distance[i][j] * x_vars[i, j] for i in facilities for j in neighborhoods)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in facilities)) == 1 for j in neighborhoods), name='')
m.addConstr(gp.quicksum((y_vars[i] for i in facilities)) == p, name='open_facilities')
m.addConstrs((x_vars[i, j] <= y_vars[i] for i in facilities for j in neighborhoods), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')