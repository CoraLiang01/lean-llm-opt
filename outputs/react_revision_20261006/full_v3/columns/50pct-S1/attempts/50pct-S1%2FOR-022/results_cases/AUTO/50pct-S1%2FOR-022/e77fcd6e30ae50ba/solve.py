CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the Superstore chain, multiple branches require inventory replenishment, and several suppliers located '
          'in different cities can provide the necessary goods. Each supplier incurs a fixed cost upon starting '
          'operations, with the fixed cost data provided in the ‚Äúfixed_cost.csv‚Äù file. Each branch needs to source '
          'a certain quantity of goods from these suppliers. For each branch, the transportation cost per unit of '
          'goods from each supplier is recorded in the ‚Äútransportation_costs.csv‚Äù file. Demand information can be '
          "gained in 'demand.csv'. The objective is to determine which suppliers to activate so that the demand of all "
          'branches is met while minimizing the total cost. The decision variables y_i are binary, indicating whether '
          'a supplier is operational (open). The decision variables x_{ij} represent the quantity of goods that branch '
          'S_j sources from supplier F_i. For each branch, x_{ij} represents the proportion of the total supply '
          'obtained from different suppliers. These decision variables help determine the optimal allocation of supply '
          'to minimize the total of fixed and transportation costs.',
 'relationships': [{'column_axis': {'id_column': 'customer_id', 'table_id': 'file_0_view_0'},
                    'column_id_mapping': {'transportation_cost_to_C1': 'C1',
                                          'transportation_cost_to_C2': 'C2',
                                          'transportation_cost_to_C3': 'C3',
                                          'transportation_cost_to_C4': 'C4',
                                          'transportation_cost_to_C5': 'C5'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'facility_id', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'facility_id',
                    'type': 'matrix'}],
 'route': 'FLP',
 'tables': [{'columns': ['customer_id', 'demand_units'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'customer_id': 'C1', 'demand_units': '143'}},
                         {'source_row': 1, 'values': {'customer_id': 'C2', 'demand_units': '6'}},
                         {'source_row': 2, 'values': {'customer_id': 'C3', 'demand_units': '10'}},
                         {'source_row': 3, 'values': {'customer_id': 'C4', 'demand_units': '25'}},
                         {'source_row': 4, 'values': {'customer_id': 'C5', 'demand_units': '3'}}],
             'returned_rows': 5,
             'role': 'branch demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['facility_id', 'fixed_opening_cost'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'facility_id': 'S1', 'fixed_opening_cost': '97.65'}},
                         {'source_row': 1, 'values': {'facility_id': 'S2', 'fixed_opening_cost': '99.76'}},
                         {'source_row': 2, 'values': {'facility_id': 'S3', 'fixed_opening_cost': '100.76'}},
                         {'source_row': 3, 'values': {'facility_id': 'S4', 'fixed_opening_cost': '105.32'}},
                         {'source_row': 4, 'values': {'facility_id': 'S5', 'fixed_opening_cost': '98.88'}}],
             'returned_rows': 5,
             'role': 'supplier fixed cost',
             'table_id': 'file_1_view_0'},
            {'columns': ['facility_id',
                         'transportation_cost_to_C1',
                         'transportation_cost_to_C2',
                         'transportation_cost_to_C3',
                         'transportation_cost_to_C4',
                         'transportation_cost_to_C5'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'facility_id': 'S1',
                                     'transportation_cost_to_C1': '150.74',
                                     'transportation_cost_to_C2': '0.02',
                                     'transportation_cost_to_C3': '49.13',
                                     'transportation_cost_to_C4': '2080.15',
                                     'transportation_cost_to_C5': '426.4'}},
                         {'source_row': 1,
                          'values': {'facility_id': 'S2',
                                     'transportation_cost_to_C1': '233.05',
                                     'transportation_cost_to_C2': '97.73',
                                     'transportation_cost_to_C3': '49.84',
                                     'transportation_cost_to_C4': '1982.39',
                                     'transportation_cost_to_C5': '23.96'}},
                         {'source_row': 2,
                          'values': {'facility_id': 'S3',
                                     'transportation_cost_to_C1': '55.68',
                                     'transportation_cost_to_C2': '935.61',
                                     'transportation_cost_to_C3': '4.03',
                                     'transportation_cost_to_C4': '73.09',
                                     'transportation_cost_to_C5': '525.32'}},
                         {'source_row': 3,
                          'values': {'facility_id': 'S4',
                                     'transportation_cost_to_C1': '1483.82',
                                     'transportation_cost_to_C2': '1801.08',
                                     'transportation_cost_to_C3': '112.16',
                                     'transportation_cost_to_C4': '816.05',
                                     'transportation_cost_to_C5': '107.01'}},
                         {'source_row': 4,
                          'values': {'facility_id': 'S5',
                                     'transportation_cost_to_C1': '1119.47',
                                     'transportation_cost_to_C2': '884.31',
                                     'transportation_cost_to_C3': '0.08',
                                     'transportation_cost_to_C4': '1544.95',
                                     'transportation_cost_to_C5': '543.67'}}],
             'returned_rows': 5,
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'unique_complete_suffix',
                                   'expected_shape': [5, 5],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [5, 5]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    demand_table = None
    fixed_cost_table = None
    cost_matrix_table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            demand_table = t
        elif t['table_id'] == 'file_1_view_0':
            fixed_cost_table = t
        elif t['table_id'] == 'file_2_view_0':
            cost_matrix_table = t
    if demand_table is None or fixed_cost_table is None or cost_matrix_table is None:
        raise RuntimeError('Missing required table(s) in CSVQA_DATA.')
    suppliers = []
    for rec in fixed_cost_table['records']:
        suppliers.append(rec['values']['facility_id'])
    customers = []
    for rec in demand_table['records']:
        customers.append(rec['values']['customer_id'])
    demand = {}
    for rec in demand_table['records']:
        cid = rec['values']['customer_id']
        try:
            demand[cid] = float(rec['values']['demand_units'])
        except Exception:
            raise ValueError(f'Invalid demand_units for customer {cid}')
    fixed_cost = {}
    for rec in fixed_cost_table['records']:
        sid = rec['values']['facility_id']
        try:
            fixed_cost[sid] = float(rec['values']['fixed_opening_cost'])
        except Exception:
            raise ValueError(f'Invalid fixed_opening_cost for supplier {sid}')
    col_map = None
    for rel in CSVQA_DATA.get('relationships', []):
        if rel.get('matrix_table_id') == 'file_2_view_0':
            col_map = rel.get('column_id_mapping')
            break
    if col_map is None:
        raise RuntimeError('Missing column_id_mapping for transportation cost matrix.')
    cost = {}
    for rec in cost_matrix_table['records']:
        sid = rec['values']['facility_id']
        cost[sid] = {}
        for (col, cid) in col_map.items():
            try:
                cost[sid][cid] = float(rec['values'][col])
            except Exception:
                raise ValueError(f'Invalid transportation cost for supplier {sid}, customer {cid}')
    if set(suppliers) != set(cost.keys()):
        raise ValueError('Mismatch in supplier keys between fixed cost and cost matrix.')
    for sid in suppliers:
        if set(customers) != set(cost[sid].keys()):
            raise ValueError(f'Mismatch in customer keys for supplier {sid} in cost matrix.')
    M = sum((demand[cid] for cid in customers))
    m = gp.Model('FLP')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)) + gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= M * y[i] for i in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()