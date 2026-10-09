CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Several suppliers located in different regions can provide the necessary Adidas products to various stores. '
          'Each supplier incurs a fixed cost when starting operations, with the fixed cost data provided in '
          '“fixed_cost.csv.” The company needs to source a unit of each Adidas product for the stores from these '
          'suppliers. For each product, the transportation cost per unit from each supplier to each store is recorded '
          'in “transportation_costs.csv.” Additionally, each store has a specific demand for these products, which is '
          'provided in “demand.csv.” The objective is to determine which suppliers to activate so that the demand for '
          'all Adidas products across all stores is met while minimizing the total cost. The decision variables y_i '
          'are binary, indicating whether a supplier is operational (open). The decision variables x_{ij} represent '
          'the quantity of goods that each store S_j sources from supplier F_i.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'FLP',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {},
             'original_rows': 6,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '216'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '216'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '216'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '144'}},
                         {'source_row': 4, 'values': {'customer': 'C5', 'demand': '144'}},
                         {'source_row': 5, 'values': {'customer': 'C6', 'demand': '144'}}],
             'returned_rows': 6,
             'role': 'store demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {},
             'original_rows': 6,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '98.88'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '99.73'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '94.01000000000001'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '93.77'}},
                         {'source_row': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '107.59'}},
                         {'source_row': 5, 'values': {'Unnamed: 0': 'S6', 'fixed_costs': '112.65'}}],
             'returned_rows': 6,
             'role': 'supplier fixed cost',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {},
             'original_rows': 6,
             'records': [{'source_row': 0,
                          'values': {'C1': '0.08',
                                     'C2': '52.33',
                                     'C3': '73.56999999999999',
                                     'C4': '1237.33',
                                     'C5': '0.07000000000000001',
                                     'C6': '112.16',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '46.02',
                                     'C2': '175.23',
                                     'C3': '2026.83',
                                     'C4': '299.89',
                                     'C5': '966.53',
                                     'C6': '1590.42',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '1031.74',
                                     'C2': '78.13',
                                     'C3': '99.02',
                                     'C4': '277.07',
                                     'C5': '884.45',
                                     'C6': '1800.86',
                                     'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'C1': '868.75',
                                     'C2': '94.2',
                                     'C3': '1776.34',
                                     'C4': '285.48',
                                     'C5': '868.85',
                                     'C6': '86.55',
                                     'Unnamed: 0': 'S4'}},
                         {'source_row': 4,
                          'values': {'C1': '1577',
                                     'C2': '760.15',
                                     'C3': '2090.19',
                                     'C4': '43.2',
                                     'C5': '1577.12',
                                     'C6': '1095.17',
                                     'Unnamed: 0': 'S5'}},
                         {'source_row': 5,
                          'values': {'C1': '49.14',
                                     'C2': '4.33',
                                     'C3': '2079.57',
                                     'C4': '277.04',
                                     'C5': '1032.01',
                                     'C6': '1543.49',
                                     'Unnamed: 0': 'S6'}}],
             'returned_rows': 6,
             'role': 'supplier-store transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [6, 6],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [6, 6]}],
                'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
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
            d_j[customer] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand for store {customer}: {row['demand']}")
    f_i = {}
    for (_, row) in fixed_cost_frame.iterrows():
        supplier = row['Unnamed: 0']
        try:
            f_i[supplier] = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Invalid fixed cost for supplier {supplier}: {row['fixed_costs']}")
    c_ij = {}
    for (_, row) in cost_frame.iterrows():
        supplier = row['Unnamed: 0']
        c_ij[supplier] = {}
        for store in J:
            if store in cost_frame.columns:
                val = row[store]
                try:
                    c_ij[supplier][store] = float(val)
                except Exception:
                    raise ValueError(f'Invalid transportation cost for supplier {supplier}, store {store}: {val}')
            else:
                continue
    for j in J:
        if j not in d_j:
            raise ValueError(f'Missing demand for store {j}')
    for i in I:
        if i not in f_i:
            raise ValueError(f'Missing fixed cost for supplier {i}')
        if i not in c_ij:
            raise ValueError(f'Missing cost row for supplier {i}')
        for j in J:
            if j not in c_ij[i]:
                raise ValueError(f'Missing transportation cost for supplier {i}, store {j}')
    M_ij = {}
    for i in I:
        M_ij[i] = {}
        for j in J:
            M_ij[i][j] = d_j[j]
    m = gp.Model('FLP_Adidas')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in I for j in J]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    open_vars = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c_ij[i][j] * quantity_vars[i, j] for (i, j) in quantity_keys)) + gp.quicksum((f_i[i] * open_vars[i] for i in I)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((quantity_vars[i, j] for i in I)) == d_j[j], name=f'demand_{j}')
    for i in I:
        for j in J:
            m.addConstr(quantity_vars[i, j] <= M_ij[i][j] * open_vars[i], name=f'activation_{i}_{j}')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')