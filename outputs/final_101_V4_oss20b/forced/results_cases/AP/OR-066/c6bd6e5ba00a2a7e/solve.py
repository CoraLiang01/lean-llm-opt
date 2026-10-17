CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Several supermarkets require inventory replenishment, and multiple suppliers located across various cities '
          'are capable of providing the needed goods. Each supplier incurs a one-time fixed cost when activated, with '
          'these costs detailed in “fixed_cost.csv.” Every supermarket must procure a unit of goods from the available '
          'suppliers. The per-unit transportation cost from each supplier to each supermarket is listed in '
          '“transportation_costs.csv,” while demand data is available in “demand.csv.” The goal is to identify which '
          'suppliers to activate in order to fulfill all supermarket demands at the lowest possible total cost. Binary '
          'decision variables y_i indicate whether supplier F_i is operational. Variables x_{ij} represent the share '
          'of supply that supermarket S_j receives from supplier F_i. These variables are used to determine the most '
          'cost-effective distribution strategy, minimizing the sum of fixed and transportation costs.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'AP',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '144'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '216'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '105.97'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '85.31'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'C1': '2358.39', 'C2': '1492.08', 'Unnamed: 0': 'S1'}},
                         {'source_row': 1, 'values': {'C1': '0.07000000000000001', 'C2': '52.32', 'Unnamed: 0': 'S2'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [2, 2],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [2, 2]}],
                'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
    f_table = tables['file_1_view_0']
    f = {}
    for rec in f_table['records']:
        supplier = rec['values']['Unnamed: 0']
        cost = float(rec['values']['fixed_costs'])
        f[supplier] = cost
    I = list(f.keys())
    d_table = tables['file_0_view_0']
    d = {}
    for rec in d_table['records']:
        customer = rec['values']['customer']
        demand = float(rec['values']['demand'])
        d[customer] = demand
    J = list(d.keys())
    c_table = tables['file_2_view_0']
    c = {}
    row_id_col = 'Unnamed: 0'
    col_ids = [col for col in c_table['columns'] if col != row_id_col]
    for rec in c_table['records']:
        supplier = rec['values'][row_id_col]
        for cust in col_ids:
            cost = float(rec['values'][cust])
            c[supplier, cust] = cost
    for i in I:
        if i not in f:
            raise ValueError(f'Missing fixed cost for supplier {i}')
    for j in J:
        if j not in d:
            raise ValueError(f'Missing demand for customer {j}')
    for i in I:
        for j in J:
            if (i, j) not in c:
                raise ValueError(f'Missing transportation cost for supplier {i} to customer {j}')
    M = sum(d.values())
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    y = model.addVars(I, vtype=GRB.BINARY, name='')
    x = model.addVars([(i, j) for i in I for j in J], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((f[i] * y[i] for i in I)) + gurobipy.quicksum((c[i, j] * x[i, j] for i in I for j in J))
    model.setObjective(obj, GRB.MINIMIZE)
    model.addConstrs((gurobipy.quicksum((x[i, j] for i in I)) == d[j] for j in J), name='')
    model.addConstrs((x[i, j] <= M * y[i] for i in I for j in J), name='')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()