CSVQA_DATA = {'query': 'Several suppliers located in different regions can provide the necessary Adidas products to various stores. Each supplier incurs a fixed cost when starting operations, with the fixed cost data provided in “fixed_cost.csv.” The company needs to source a unit of each Adidas product for the stores from these suppliers. For each product, the transportation cost per unit from each supplier to each store is recorded in “transportation_costs.csv.” Additionally, each store has a specific demand for these products, which is provided in “demand.csv.” The objective is to determine which suppliers to activate so that the demand for all Adidas products across all stores is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a supplier is operational (open). The decision variables x_{ij} represent the quantity of goods that each store S_j sources from supplier F_i.', 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'}, 'matrix_table_id': 'file_2_view_0', 'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Unnamed: 0', 'type': 'matrix'}], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '216'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '216'}}, {'data_row_number': 2, 'values': {'customer': 'C3', 'demand': '216'}}, {'data_row_number': 3, 'values': {'customer': 'C4', 'demand': '144'}}, {'data_row_number': 4, 'values': {'customer': 'C5', 'demand': '144'}}, {'data_row_number': 5, 'values': {'customer': 'C6', 'demand': '144'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'fixed_costs'], 'file_name': 'fixed_cost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '98.88'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '99.73'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '94.01000000000001'}}, {'data_row_number': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '93.77'}}, {'data_row_number': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '107.59'}}, {'data_row_number': 5, 'values': {'Unnamed: 0': 'S6', 'fixed_costs': '112.65'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '0.08', 'C2': '52.33', 'C3': '73.56999999999999', 'C4': '1237.33', 'C5': '0.07000000000000001', 'C6': '112.16', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '46.02', 'C2': '175.23', 'C3': '2026.83', 'C4': '299.89', 'C5': '966.53', 'C6': '1590.42', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'C1': '1031.74', 'C2': '78.13', 'C3': '99.02', 'C4': '277.07', 'C5': '884.45', 'C6': '1800.86', 'Unnamed: 0': 'S3'}}, {'data_row_number': 3, 'values': {'C1': '868.75', 'C2': '94.2', 'C3': '1776.34', 'C4': '285.48', 'C5': '868.85', 'C6': '86.55', 'Unnamed: 0': 'S4'}}, {'data_row_number': 4, 'values': {'C1': '1577', 'C2': '760.15', 'C3': '2090.19', 'C4': '43.2', 'C5': '1577.12', 'C6': '1095.17', 'Unnamed: 0': 'S5'}}, {'data_row_number': 5, 'values': {'C1': '49.14', 'C2': '4.33', 'C3': '2079.57', 'C4': '277.04', 'C5': '1032.01', 'C6': '1543.49', 'Unnamed: 0': 'S6'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def get_table(table_id):
    for tbl in CSVQA_DATA['tables']:
        if tbl['table_id'] == table_id:
            return tbl
    raise ValueError(f'Table {table_id} not found in CSVQA_DATA')

def solve_problem():
    demand_tbl = get_table('file_0_view_0')
    fixed_tbl = get_table('file_1_view_0')
    trans_tbl = get_table('file_2_view_0')
    I = set()
    J = set()
    f = {}
    for rec in fixed_tbl['records']:
        sup = rec['values']['Unnamed: 0']
        cost_str = rec['values']['fixed_costs']
        try:
            cost = float(cost_str)
        except ValueError:
            raise ValueError(f'Invalid fixed cost for supplier {sup}')
        f[sup] = cost
        I.add(sup)
    d = {}
    for rec in demand_tbl['records']:
        store = rec['values']['customer']
        dem_str = rec['values']['demand']
        try:
            dem = float(dem_str)
        except ValueError:
            raise ValueError(f'Invalid demand for store {store}')
        d[store] = dem
        J.add(store)
    t = {}
    for rec in trans_tbl['records']:
        sup = rec['values']['Unnamed: 0']
        if sup not in I:
            raise ValueError(f'Supplier {sup} in transportation table not in fixed cost table')
        t[sup] = {}
        for store in J:
            if store not in rec['values']:
                raise ValueError(f'Missing column {store} in transportation table for supplier {sup}')
            val_str = rec['values'][store]
            try:
                val = float(val_str)
            except ValueError:
                raise ValueError(f'Invalid transportation cost for supplier {sup} to store {store}')
            t[sup][store] = val
    for sup in I:
        if sup not in t:
            raise ValueError(f'Missing transportation data for supplier {sup}')
        if len(t[sup]) != len(J):
            raise ValueError(f'Incomplete transportation data for supplier {sup}')
    for store in J:
        if store not in d:
            raise ValueError(f'Missing demand data for store {store}')
    M = sum((d[store] for store in J))
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    y = model.addVars(I, vtype=GRB.BINARY, name='')
    x = model.addVars([(i, j) for i in I for j in J], vtype=GRB.CONTINUOUS, name='')
    obj = gurobipy.quicksum((f[i] * y[i] for i in I))
    obj += gurobipy.quicksum((t[i][j] * x[i, j] for i in I for j in J))
    model.setObjective(obj, GRB.MINIMIZE)
    for j in J:
        model.addConstr(gurobipy.quicksum((x[i, j] for i in I)) == d[j], name=f'demand_{j}')
    for i in I:
        for j in J:
            model.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
    model.optimize()
    return model
m = solve_problem()
