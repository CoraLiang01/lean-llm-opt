CSVQA_DATA = {'query': 'The company must select an optimal subset of potential warehouse locations to serve all customers. The objective is to minimize total cost, which includes the fixed annual opening costs of each selected warehouse and the variable transportation costs for fulfilling all customer demand. Each potential warehouse has a maximum service capacity. Relevant information is provided in the tables.', 'relationships': [{'column_axis': {'id_column': 'Customer ID', 'table_id': 'file_2_view_0'}, 'matrix_table_id': 'file_0_view_0', 'row_axis': {'id_column': 'Warehouse ID', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Warehouse ID', 'type': 'matrix'}], 'tables': [{'columns': ['Warehouse ID', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20'], 'file_name': 'cost.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '10', 'C10': '9', 'C11': '14', 'C12': '19', 'C13': '25', 'C14': '13', 'C15': '17', 'C16': '6', 'C17': '21', 'C18': '15', 'C19': '8', 'C2': '15', 'C20': '10', 'C3': '20', 'C4': '11', 'C5': '16', 'C6': '18', 'C7': '7', 'C8': '12', 'C9': '22', 'Warehouse ID': 'W1'}}, {'data_row_number': 1, 'values': {'C1': '18', 'C10': '16', 'C11': '20', 'C12': '8', 'C13': '15', 'C14': '22', 'C15': '7', 'C16': '13', 'C17': '24', 'C18': '17', 'C19': '12', 'C2': '12', 'C20': '6', 'C3': '9', 'C4': '14', 'C5': '10', 'C6': '5', 'C7': '19', 'C8': '23', 'C9': '11', 'Warehouse ID': 'W2'}}, {'data_row_number': 2, 'values': {'C1': '13', 'C10': '24', 'C11': '13', 'C12': '22', 'C13': '7', 'C14': '19', 'C15': '14', 'C16': '18', 'C17': '9', 'C18': '25', 'C19': '11', 'C2': '17', 'C20': '16', 'C3': '15', 'C4': '8', 'C5': '12', 'C6': '21', 'C7': '16', 'C8': '10', 'C9': '5', 'Warehouse ID': 'W3'}}, {'data_row_number': 3, 'values': {'C1': '7', 'C10': '25', 'C11': '6', 'C12': '14', 'C13': '21', 'C14': '9', 'C15': '23', 'C16': '17', 'C17': '10', 'C18': '18', 'C19': '24', 'C2': '22', 'C20': '5', 'C3': '11', 'C4': '16', 'C5': '20', 'C6': '8', 'C7': '15', 'C8': '19', 'C9': '13', 'Warehouse ID': 'W4'}}, {'data_row_number': 4, 'values': {'C1': '16', 'C10': '21', 'C11': '5', 'C12': '17', 'C13': '9', 'C14': '24', 'C15': '12', 'C16': '20', 'C17': '6', 'C18': '15', 'C19': '19', 'C2': '9', 'C20': '11', 'C3': '25', 'C4': '13', 'C5': '7', 'C6': '10', 'C7': '23', 'C8': '14', 'C9': '18', 'Warehouse ID': 'W5'}}, {'data_row_number': 5, 'values': {'C1': '22', 'C10': '12', 'C11': '15', 'C12': '24', 'C13': '5', 'C14': '20', 'C15': '10', 'C16': '25', 'C17': '13', 'C18': '7', 'C19': '18', 'C2': '6', 'C20': '16', 'C3': '14', 'C4': '19', 'C5': '23', 'C6': '11', 'C7': '8', 'C8': '17', 'C9': '9', 'Warehouse ID': 'W6'}}, {'data_row_number': 6, 'values': {'C1': '8', 'C10': '20', 'C11': '18', 'C12': '13', 'C13': '24', 'C14': '5', 'C15': '19', 'C16': '12', 'C17': '23', 'C18': '10', 'C19': '7', 'C2': '25', 'C20': '15', 'C3': '17', 'C4': '9', 'C5': '14', 'C6': '22', 'C7': '11', 'C8': '6', 'C9': '16', 'Warehouse ID': 'W7'}}, {'data_row_number': 7, 'values': {'C1': '19', 'C10': '8', 'C11': '17', 'C12': '10', 'C13': '12', 'C14': '23', 'C15': '5', 'C16': '14', 'C17': '22', 'C18': '9', 'C19': '16', 'C2': '11', 'C20': '25', 'C3': '7', 'C4': '21', 'C5': '15', 'C6': '24', 'C7': '13', 'C8': '16', 'C9': '20', 'Warehouse ID': 'W8'}}, {'data_row_number': 8, 'values': {'C1': '12', 'C10': '11', 'C11': '16', 'C12': '21', 'C13': '10', 'C14': '7', 'C15': '24', 'C16': '15', 'C17': '19', 'C18': '6', 'C19': '13', 'C2': '20', 'C20': '22', 'C3': '5', 'C4': '23', 'C5': '17', 'C6': '14', 'C7': '9', 'C8': '25', 'C9': '18', 'Warehouse ID': 'W9'}}, {'data_row_number': 9, 'values': {'C1': '25', 'C10': '17', 'C11': '23', 'C12': '6', 'C13': '16', 'C14': '10', 'C15': '20', 'C16': '9', 'C17': '18', 'C18': '11', 'C19': '25', 'C2': '14', 'C20': '14', 'C3': '22', 'C4': '5', 'C5': '19', 'C6': '12', 'C7': '24', 'C8': '7', 'C9': '15', 'Warehouse ID': 'W10'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Warehouse ID', 'Fixed_Cost', 'Capacity'], 'file_name': 'warehouse.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '1000', 'Fixed_Cost': '2000', 'Warehouse ID': 'W1'}}, {'data_row_number': 1, 'values': {'Capacity': '1500', 'Fixed_Cost': '2500', 'Warehouse ID': 'W2'}}, {'data_row_number': 2, 'values': {'Capacity': '1200', 'Fixed_Cost': '1800', 'Warehouse ID': 'W3'}}, {'data_row_number': 3, 'values': {'Capacity': '2000', 'Fixed_Cost': '3200', 'Warehouse ID': 'W4'}}, {'data_row_number': 4, 'values': {'Capacity': '800', 'Fixed_Cost': '1500', 'Warehouse ID': 'W5'}}, {'data_row_number': 5, 'values': {'Capacity': '2500', 'Fixed_Cost': '4000', 'Warehouse ID': 'W6'}}, {'data_row_number': 6, 'values': {'Capacity': '1800', 'Fixed_Cost': '2800', 'Warehouse ID': 'W7'}}, {'data_row_number': 7, 'values': {'Capacity': '1100', 'Fixed_Cost': '1950', 'Warehouse ID': 'W8'}}, {'data_row_number': 8, 'values': {'Capacity': '2100', 'Fixed_Cost': '3500', 'Warehouse ID': 'W9'}}, {'data_row_number': 9, 'values': {'Capacity': '1300', 'Fixed_Cost': '2200', 'Warehouse ID': 'W10'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Customer ID', 'Demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'Customer ID': 'C1', 'Demand': '800'}}, {'data_row_number': 1, 'values': {'Customer ID': 'C2', 'Demand': '600'}}, {'data_row_number': 2, 'values': {'Customer ID': 'C3', 'Demand': '500'}}, {'data_row_number': 3, 'values': {'Customer ID': 'C4', 'Demand': '700'}}, {'data_row_number': 4, 'values': {'Customer ID': 'C5', 'Demand': '450'}}, {'data_row_number': 5, 'values': {'Customer ID': 'C6', 'Demand': '950'}}, {'data_row_number': 6, 'values': {'Customer ID': 'C7', 'Demand': '350'}}, {'data_row_number': 7, 'values': {'Customer ID': 'C8', 'Demand': '850'}}, {'data_row_number': 8, 'values': {'Customer ID': 'C9', 'Demand': '400'}}, {'data_row_number': 9, 'values': {'Customer ID': 'C10', 'Demand': '750'}}, {'data_row_number': 10, 'values': {'Customer ID': 'C11', 'Demand': '900'}}, {'data_row_number': 11, 'values': {'Customer ID': 'C12', 'Demand': '550'}}, {'data_row_number': 12, 'values': {'Customer ID': 'C13', 'Demand': '650'}}, {'data_row_number': 13, 'values': {'Customer ID': 'C14', 'Demand': '820'}}, {'data_row_number': 14, 'values': {'Customer ID': 'C15', 'Demand': '480'}}, {'data_row_number': 15, 'values': {'Customer ID': 'C16', 'Demand': '920'}}, {'data_row_number': 16, 'values': {'Customer ID': 'C17', 'Demand': '320'}}, {'data_row_number': 17, 'values': {'Customer ID': 'C18', 'Demand': '780'}}, {'data_row_number': 18, 'values': {'Customer ID': 'C19', 'Demand': '520'}}, {'data_row_number': 19, 'values': {'Customer ID': 'C20', 'Demand': '680'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
    wh_tbl = tables['file_1_view_0']
    wh_records = wh_tbl['records']
    f_w = {}
    cap_w = {}
    for rec in wh_records:
        vals = rec['values']
        wid = vals['Warehouse ID']
        f_w[wid] = float(vals['Fixed_Cost'])
        cap_w[wid] = float(vals['Capacity'])
    cust_tbl = tables['file_2_view_0']
    cust_records = cust_tbl['records']
    d_c = {}
    for rec in cust_records:
        vals = rec['values']
        cid = vals['Customer ID']
        d_c[cid] = float(vals['Demand'])
    cost_tbl = tables['file_0_view_0']
    cost_records = cost_tbl['records']
    cust_ids_from_cost = [col for col in cost_tbl['columns'] if col != 'Warehouse ID']
    c_wc = {}
    for rec in cost_records:
        vals = rec['values']
        wid = vals['Warehouse ID']
        for cid in cust_ids_from_cost:
            c_wc[wid, cid] = float(vals[cid])
    W = list(f_w.keys())
    C = list(d_c.keys())
    if set(W) != set(c_wc.keys()) & set(W):
        missing = set(W) - set(c_wc.keys())
        if missing:
            raise ValueError(f'Missing cost entries for warehouses: {missing}')
    if set(C) != set(c_wc.keys()) & set(C):
        missing = set(C) - set(c_wc.keys())
        if missing:
            raise ValueError(f'Missing cost entries for customers: {missing}')
    for w in W:
        if w not in f_w or w not in cap_w:
            raise ValueError(f'Missing parameters for warehouse {w}')
    for c in C:
        if c not in d_c:
            raise ValueError(f'Missing demand for customer {c}')
    m = gp.Model()
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(W, vtype=GRB.BINARY, name='')
    x = m.addVars([(w, c) for w in W for c in C], vtype=GRB.CONTINUOUS, lb=0.0, name='')
    obj = gp.quicksum((f_w[w] * y[w] for w in W)) + gp.quicksum((c_wc[w, c] * x[w, c] for w in W for c in C))
    m.setObjective(obj, GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[w, c] for w in W)) == d_c[c] for c in C), name='')
    m.addConstrs((gp.quicksum((x[w, c] for c in C)) <= cap_w[w] * y[w] for w in W), name='')
    m.optimize()
    return m
m = solve_problem()
