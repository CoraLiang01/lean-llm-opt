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
 'route': 'TP',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '144'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '216'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '105.97'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '85.31'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {},
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
    data = CSVQA_DATA
    tables = data['tables']
    demand_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    fixed_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    cost_table = next((t for t in tables if t['table_id'] == 'file_2_view_0'))
    demand = {}
    for rec in demand_table['records']:
        key = rec['values']['customer']
        demand[key] = float(rec['values']['demand'])
    fixed_cost = {}
    for rec in fixed_table['records']:
        key = rec['values']['Unnamed: 0']
        fixed_cost[key] = float(rec['values']['fixed_costs'])
    cost = {}
    for rec in cost_table['records']:
        supplier = rec['values']['Unnamed: 0']
        cost[supplier] = {}
        for col, val in rec['values'].items():
            if col == 'Unnamed: 0':
                continue
            cost[supplier][col] = float(val)
    I = set(fixed_cost.keys())
    J = set(demand.keys())
    for i in I:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {i}')
    for j in J:
        if j not in demand:
            raise ValueError(f'Missing demand for supermarket {j}')
    for i in I:
        for j in J:
            if j not in cost.get(i, {}):
                raise ValueError(f'Missing transportation cost for supplier {i} to supermarket {j}')
    M = sum(demand.values())
    m = gurobipy.Model('SupplierActivation')
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((fixed_cost[i] * y[i] for i in I)) + gurobipy.quicksum((cost[i][j] * x[i, j] for i in I for j in J))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in J:
        expr = gurobipy.quicksum((x[i, j] for i in I))
        m.addConstr(expr >= demand[j], name='')
    for i in I:
        for j in J:
            m.addConstr(x[i, j] <= M * y[i], name='')
    m.optimize()
    return m
m = solve_problem()