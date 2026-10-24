CSVQA_DATA = {'query': 'Several suppliers located in different regions can provide the necessary Adidas products to various stores. Each supplier incurs a fixed cost when starting operations, with the fixed cost data provided in “fixed_cost.csv.” The company needs to source a unit of each Adidas product for the stores from these suppliers. For each product, the transportation cost per unit from each supplier to each store is recorded in “transportation_costs.csv.” Additionally, each store has a specific demand for these products, which is provided in “demand.csv.” The objective is to determine which suppliers to activate so that the demand for all Adidas products across all stores is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a supplier is operational (open). The decision variables x_{ij} represent the quantity of goods that each store S_j sources from supplier F_i.', 'relationships': [], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '216'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '216'}}, {'data_row_number': 2, 'values': {'customer': 'C3', 'demand': '216'}}, {'data_row_number': 3, 'values': {'customer': 'C4', 'demand': '144'}}, {'data_row_number': 4, 'values': {'customer': 'C5', 'demand': '144'}}, {'data_row_number': 5, 'values': {'customer': 'C6', 'demand': '144'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'fixed_costs'], 'file_name': 'fixed_cost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '98.88'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '99.73'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '94.01000000000001'}}, {'data_row_number': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '93.77'}}, {'data_row_number': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '107.59'}}, {'data_row_number': 5, 'values': {'Unnamed: 0': 'S6', 'fixed_costs': '112.65'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '0.08', 'C2': '52.33', 'C3': '73.56999999999999', 'C4': '1237.33', 'C5': '0.07000000000000001', 'C6': '112.16', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '46.02', 'C2': '175.23', 'C3': '2026.83', 'C4': '299.89', 'C5': '966.53', 'C6': '1590.42', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'C1': '1031.74', 'C2': '78.13', 'C3': '99.02', 'C4': '277.07', 'C5': '884.45', 'C6': '1800.86', 'Unnamed: 0': 'S3'}}, {'data_row_number': 3, 'values': {'C1': '868.75', 'C2': '94.2', 'C3': '1776.34', 'C4': '285.48', 'C5': '868.85', 'C6': '86.55', 'Unnamed: 0': 'S4'}}, {'data_row_number': 4, 'values': {'C1': '1577', 'C2': '760.15', 'C3': '2090.19', 'C4': '43.2', 'C5': '1577.12', 'C6': '1095.17', 'Unnamed: 0': 'S5'}}, {'data_row_number': 5, 'values': {'C1': '49.14', 'C2': '4.33', 'C3': '2079.57', 'C4': '277.04', 'C5': '1032.01', 'C6': '1543.49', 'Unnamed: 0': 'S6'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']

    def get_table(table_id):
        for t in tables:
            if t['table_id'] == table_id:
                return t
        raise ValueError(f'Table {table_id} not found')
    table_S = get_table('file_1_view_0')
    S = [rec['values']['Unnamed: 0'] for rec in table_S['records']]
    table_C = get_table('file_0_view_0')
    C = [rec['values']['customer'] for rec in table_C['records']]
    f_s = {}
    for rec in table_S['records']:
        key = rec['values']['Unnamed: 0']
        val = rec['values']['fixed_costs']
        f_s[key] = float(val)
    d_c = {}
    for rec in table_C['records']:
        key = rec['values']['customer']
        val = rec['values']['demand']
        d_c[key] = float(val)
    table_T = get_table('file_2_view_0')
    t_s_c = {}
    for rec in table_T['records']:
        s = rec['values']['Unnamed: 0']
        for c in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']:
            val = rec['values'][c]
            t_s_c[s, c] = float(val)
    for s in S:
        if s not in f_s:
            raise ValueError(f'Missing fixed cost for supplier {s}')
    for c in C:
        if c not in d_c:
            raise ValueError(f'Missing demand for customer {c}')
    for s in S:
        for c in C:
            if (s, c) not in t_s_c:
                raise ValueError(f'Missing transportation cost for supplier {s} to customer {c}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    y = model.addVars(S, vtype=GRB.BINARY, name='')
    x = model.addVars([(s, c) for s in S for c in C], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((f_s[s] * y[s] for s in S)) + gurobipy.quicksum((t_s_c[s, c] * x[s, c] for s in S for c in C))
    model.setObjective(obj, GRB.MINIMIZE)
    model.addConstrs((gurobipy.quicksum((x[s, c] for s in S)) >= d_c[c] for c in C), name='')
    model.addConstrs((x[s, c] <= d_c[c] * y[s] for s in S for c in C), name='')
    model.optimize()
    return model
m = solve_problem()
