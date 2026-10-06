CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Several supermarkets require inventory replenishment, and multiple suppliers located across various cities '
          'are capable of providing the needed goods. Each supplier incurs a one-time fixed cost when activated, with '
          'these costs detailed in ‚Äúfixed_cost.csv.‚Äù Every supermarket must procure a unit of goods from the '
          'available suppliers. The per-unit transportation cost from each supplier to each supermarket is listed in '
          '‚Äútransportation_costs.csv,‚Äù while demand data is available in ‚Äúdemand.csv.‚Äù The goal is to identify '
          'which suppliers to activate in order to fulfill all supermarket demands at the lowest possible total cost. '
          'Binary decision variables y_i indicate whether supplier F_i is operational. Variables x_{ij} represent the '
          'share of supply that supermarket S_j receives from supplier F_i. These variables are used to determine the '
          'most cost-effective distribution strategy, minimizing the sum of fixed and transportation costs.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'FLP',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '144'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '216'}}],
             'returned_rows': 2,
             'role': 'supermarket demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '105.97'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '85.31'}}],
             'returned_rows': 2,
             'role': 'supplier fixed cost',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'C1': '2358.39', 'C2': '1492.08', 'Unnamed: 0': 'S1'}},
                         {'source_row': 1, 'values': {'C1': '0.07000000000000001', 'C2': '52.32', 'Unnamed: 0': 'S2'}}],
             'returned_rows': 2,
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [2, 2],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [2, 2]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    suppliers = []
    supplier_set = set()
    for rec in CSVQA_DATA['tables']:
        if rec['table_id'] == 'file_1_view_0':
            for r in rec['records']:
                supplier_set.add(r['values']['Unnamed: 0'])
        if rec['table_id'] == 'file_2_view_0':
            for r in rec['records']:
                supplier_set.add(r['values']['Unnamed: 0'])
    suppliers = list(supplier_set)
    suppliers.sort()
    supermarket_set = set()
    for rec in CSVQA_DATA['tables']:
        if rec['table_id'] == 'file_0_view_0':
            for r in rec['records']:
                supermarket_set.add(r['values']['customer'])
        if rec['table_id'] == 'file_2_view_0':
            for c in rec['columns']:
                if c != 'Unnamed: 0':
                    supermarket_set.add(c)
    supermarkets = list(supermarket_set)
    supermarkets.sort()
    demand = {}
    for rec in CSVQA_DATA['tables']:
        if rec['table_id'] == 'file_0_view_0':
            for r in rec['records']:
                j = r['values']['customer']
                d = float(r['values']['demand'])
                if j in demand:
                    demand[j] += d
                else:
                    demand[j] = d
    for j in supermarkets:
        if j not in demand:
            demand[j] = 0.0
    fixed_cost = {}
    for rec in CSVQA_DATA['tables']:
        if rec['table_id'] == 'file_1_view_0':
            for r in rec['records']:
                i = r['values']['Unnamed: 0']
                f = float(r['values']['fixed_costs'])
                if i in fixed_cost:
                    fixed_cost[i] += f
                else:
                    fixed_cost[i] = f
    for i in suppliers:
        if i not in fixed_cost:
            fixed_cost[i] = 0.0
    cost = {}
    for rec in CSVQA_DATA['tables']:
        if rec['table_id'] == 'file_2_view_0':
            for r in rec['records']:
                i = r['values']['Unnamed: 0']
                if i not in cost:
                    cost[i] = {}
                for j in supermarkets:
                    if j in r['values']:
                        c = float(r['values'][j])
                        cost[i][j] = c
    for i in suppliers:
        if i not in cost:
            raise ValueError(f'Missing cost row for supplier {i}')
        for j in supermarkets:
            if j not in cost[i]:
                raise ValueError(f'Missing cost for supplier {i}, supermarket {j}')
    M = sum((demand[j] for j in supermarkets))
    m = gp.Model('FLP')
    m.setParam('MIPGap', 0.0001)
    x_keys = [(i, j) for i in suppliers for j in supermarkets]
    x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in supermarkets)) + gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in supermarkets), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in supermarkets)) <= M * y[i] for i in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()