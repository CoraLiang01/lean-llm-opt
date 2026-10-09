CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the context of Colorado Motor Vehicle Sales, multiple car dealerships require inventory, and several '
          'suppliers located in different cities can provide the necessary vehicles. Each supplier incurs a fixed cost '
          'upon starting operations, with the fixed cost data provided in “fixed_cost.csv.” Each dealership needs to '
          'source a certain number of vehicles from these suppliers. For each dealership, the transportation cost per '
          'vehicle from each supplier is recorded in “transportation_costs.csv.” Demand information can be gained in '
          "'demand.csv'. The objective is to determine which suppliers to open so that the demand of all dealerships "
          'is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a '
          'supplier is operational (open). The decision variables x_{ij} represent the quantity of vehicles that '
          'dealership S_j sources from supplier F_i. For each dealership, x_{ij} represents the proportion of the '
          'total supply obtained from different suppliers.',
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
             'original_rows': 9,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '4742532000'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '1600594000'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '5086889000'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '1027326000'}},
                         {'source_row': 4, 'values': {'customer': 'C5', 'demand': '11926044000'}},
                         {'source_row': 5, 'values': {'customer': 'C6', 'demand': '9058407000'}},
                         {'source_row': 6, 'values': {'customer': 'C7', 'demand': '5344367000'}},
                         {'source_row': 7, 'values': {'customer': 'C8', 'demand': '677201000'}},
                         {'source_row': 8, 'values': {'customer': 'C9', 'demand': '3236493000'}}],
             'returned_rows': 9,
             'role': 'dealership demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '100.64'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '98.72'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '100.18'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '96.58'}},
                         {'source_row': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '95.75'}},
                         {'source_row': 5, 'values': {'Unnamed: 0': 'S6', 'fixed_costs': '99.06'}},
                         {'source_row': 6, 'values': {'Unnamed: 0': 'S7', 'fixed_costs': '101.78'}},
                         {'source_row': 7, 'values': {'Unnamed: 0': 'S8', 'fixed_costs': '93.86'}}],
             'returned_rows': 8,
             'role': 'supplier fixed cost',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0,
                          'values': {'C1': '1091.04',
                                     'C2': '85.72',
                                     'C3': '99.08',
                                     'C4': '747.35',
                                     'C5': '893.86',
                                     'C6': '23.65',
                                     'C7': '15.11',
                                     'C8': '15.03',
                                     'C9': '497.88',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '58.88',
                                     'C2': '1617.16',
                                     'C3': '1786.44',
                                     'C4': '951.81',
                                     'C5': '56.45',
                                     'C6': '642.77',
                                     'C7': '16.69',
                                     'C8': '0.63',
                                     'C9': '11.2',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '110.47',
                                     'C2': '0.04',
                                     'C3': '38.89',
                                     'C4': '1397.95',
                                     'C5': '2361.45',
                                     'C6': '107.62',
                                     'C7': '1598.5',
                                     'C8': '76.41',
                                     'C9': '1382.84',
                                     'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'C1': '1458.85',
                                     'C2': '1049.27',
                                     'C3': '597.32',
                                     'C4': '1731.9',
                                     'C5': '69.09',
                                     'C6': '1227.17',
                                     'C7': '1187.55',
                                     'C8': '1017.16',
                                     'C9': '52.15',
                                     'Unnamed: 0': 'S4'}},
                         {'source_row': 4,
                          'values': {'C1': '0.38',
                                     'C2': '2315.52',
                                     'C3': '1313.06',
                                     'C4': '1253.71',
                                     'C5': '50.24',
                                     'C6': '29.19',
                                     'C7': '60.17',
                                     'C8': '1077.35',
                                     'C9': '70.11',
                                     'Unnamed: 0': 'S5'}},
                         {'source_row': 5,
                          'values': {'C1': '58.2',
                                     'C2': '1395.81',
                                     'C3': '84.6',
                                     'C4': '830.64',
                                     'C5': '1003.86',
                                     'C6': '631.17',
                                     'C7': '31.13',
                                     'C8': '1.4',
                                     'C9': '246.24',
                                     'Unnamed: 0': 'S6'}},
                         {'source_row': 6,
                          'values': {'C1': '1255.23',
                                     'C2': '1382.31',
                                     'C3': '78.79',
                                     'C4': '829.02',
                                     'C5': '67.31',
                                     'C6': '877.35',
                                     'C7': '185.28',
                                     'C8': '221.98',
                                     'C9': '0.05',
                                     'Unnamed: 0': 'S7'}},
                         {'source_row': 7,
                          'values': {'C1': '1990.09',
                                     'C2': '1.23',
                                     'C3': '38.97',
                                     'C4': '1396.35',
                                     'C5': '112.54',
                                     'C6': '107.54',
                                     'C7': '1596.74',
                                     'C8': '76.32',
                                     'C9': '1183.79',
                                     'Unnamed: 0': 'S8'}}],
             'returned_rows': 8,
             'role': 'supplier-dealership transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [8, 9],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [8, 9]}],
                'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    demand_frame = CSVQA_FRAMES['file_0_view_0']
    fixed_cost_frame = CSVQA_FRAMES['file_1_view_0']
    cost_frame = CSVQA_FRAMES['file_2_view_0']
    I = []
    for (_, row) in fixed_cost_frame.iterrows():
        supplier = row['Unnamed: 0']
        I.append(supplier)
    for (_, row) in cost_frame.iterrows():
        supplier = row['Unnamed: 0']
        if supplier not in I:
            I.append(supplier)
    J = []
    for (_, row) in demand_frame.iterrows():
        customer = row['customer']
        J.append(customer)
    for col in cost_frame.columns:
        if col != 'Unnamed: 0' and col not in J:
            J.append(col)
    d_j = {}
    for (_, row) in demand_frame.iterrows():
        customer = row['customer']
        try:
            demand = float(row['demand'])
        except Exception:
            raise ValueError(f'Non-numeric demand for customer {customer}')
        d_j[customer] = demand
    f_i = {}
    for (_, row) in fixed_cost_frame.iterrows():
        supplier = row['Unnamed: 0']
        try:
            fixed_cost = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f'Non-numeric fixed cost for supplier {supplier}')
        f_i[supplier] = fixed_cost
    c_ij = {}
    for (_, row) in cost_frame.iterrows():
        supplier = row['Unnamed: 0']
        c_ij[supplier] = {}
        for customer in J:
            if customer == 'Unnamed: 0':
                continue
            if customer in row:
                val = row[customer]
            else:
                val = row.get(customer, '')
            try:
                cost = float(val)
            except Exception:
                raise ValueError(f'Non-numeric transportation cost for supplier {supplier}, customer {customer}')
            c_ij[supplier][customer] = cost
    M = sum((d_j[j] for j in J if j in d_j))
    M_i = {i: M for i in I}
    for i in I:
        if i not in f_i:
            raise ValueError(f'Missing fixed cost for supplier {i}')
        if i not in c_ij:
            raise ValueError(f'Missing cost row for supplier {i}')
        for j in J:
            if j not in c_ij[i]:
                raise ValueError(f'Missing transportation cost for supplier {i}, customer {j}')
    for j in J:
        if j not in d_j:
            raise ValueError(f'Missing demand for customer {j}')
    m = gp.Model('ColoradoMotorVehicleFLP')
    m.setParam('MIPGap', 0.0001)
    x_vars = m.addVars(I, J, lb=0, vtype=GRB.CONTINUOUS, name='')
    y_vars = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c_ij[i][j] * x_vars[i, j] for i in I for j in J)) + gp.quicksum((f_i[i] * y_vars[i] for i in I)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in I)) == d_j[j] for j in J), name='')
    m.addConstrs((gp.quicksum((x_vars[i, j] for j in J)) <= M_i[i] * y_vars[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')