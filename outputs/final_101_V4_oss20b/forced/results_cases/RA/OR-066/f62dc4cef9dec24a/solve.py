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
 'route': 'RA',
 'tables': [{'aliases': ['file_0_view_0', 'demand.csv'],
             'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '144'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '216'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'aliases': ['file_1_view_0', 'fixed_cost.csv'],
             'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '105.97'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '85.31'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'aliases': ['file_2_view_0', 'transportation_costs.csv'],
             'columns': ['Unnamed: 0', 'C1', 'C2'],
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
                                   'expected_shape': [2, 2],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'shape': [2, 2]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
demand = {}
fixed_cost = {}
transport_cost = {}
for table in CSVQA_DATA['tables']:
    if table['table_id'] == 'file_0_view_0' or 'demand.csv' in table.get('aliases', []):
        for rec in table['records']:
            cust = rec['values']['customer']
            demand[cust] = int(rec['values']['demand'])
        break
for table in CSVQA_DATA['tables']:
    if table['table_id'] == 'file_1_view_0' or 'fixed_cost.csv' in table.get('aliases', []):
        for rec in table['records']:
            sup = rec['values']['Unnamed: 0']
            fixed_cost[sup] = float(rec['values']['fixed_costs'])
        break
for table in CSVQA_DATA['tables']:
    if table['table_id'] == 'file_2_view_0' or 'transportation_costs.csv' in table.get('aliases', []):
        for rec in table['records']:
            sup = rec['values']['Unnamed: 0']
            for cust in demand.keys():
                transport_cost[sup, cust] = float(rec['values'][cust])
        break
m = gp.Model('facility_location')
y = {}
for sup in fixed_cost:
    y[sup] = m.addVar(vtype=GRB.BINARY)
x = {}
for sup in fixed_cost:
    for cust in demand:
        x[sup, cust] = m.addVar(vtype=GRB.INTEGER)
m.update()
for cust in demand:
    m.addConstr(gp.quicksum((x[sup, cust] for sup in fixed_cost)) >= demand[cust])
for sup in fixed_cost:
    for cust in demand:
        m.addConstr(x[sup, cust] <= demand[cust] * y[sup])
obj = gp.quicksum((fixed_cost[sup] * y[sup] for sup in fixed_cost)) + gp.quicksum((transport_cost[sup, cust] * x[sup, cust] for sup in fixed_cost for cust in demand))
m.setObjective(obj, GRB.MINIMIZE)
m.Params.MIPGap = 1e-06
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print(f'Solver status: {m.Status}')