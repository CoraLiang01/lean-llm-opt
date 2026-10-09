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
             'role': 'supplier fixed costs',
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
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    suppliers = []
    fixed_cost = {}
    fixed_cost_frame = CSVQA_FRAMES['file_1_view_0']
    for (_, row) in fixed_cost_frame.iterrows():
        supplier = row['Unnamed: 0']
        suppliers.append(supplier)
        try:
            fixed_cost[supplier] = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Invalid fixed_costs for supplier {supplier}: {row['fixed_costs']}")
    supermarkets = []
    demand = {}
    demand_frame = CSVQA_FRAMES['file_0_view_0']
    for (_, row) in demand_frame.iterrows():
        customer = row['customer']
        supermarkets.append(customer)
        try:
            demand[customer] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand for customer {customer}: {row['demand']}")
    cost = {}
    cost_frame = CSVQA_FRAMES['file_2_view_0']
    for (_, row) in cost_frame.iterrows():
        supplier = row['Unnamed: 0']
        if supplier not in suppliers:
            raise ValueError(f'Supplier {supplier} in transportation_costs.csv not found in fixed_cost.csv')
        cost[supplier] = {}
        for customer in supermarkets:
            if customer not in row:
                raise ValueError(f'Customer {customer} not found in transportation_costs.csv columns')
            try:
                cost[supplier][customer] = float(row[customer])
            except Exception:
                raise ValueError(f'Invalid transportation cost for ({supplier},{customer}): {row[customer]}')
    if set(cost.keys()) != set(suppliers):
        raise ValueError('Mismatch in supplier keys between cost matrix and fixed_costs')
    for supplier in suppliers:
        if set(cost[supplier].keys()) != set(supermarkets):
            raise ValueError(f'Mismatch in supermarket keys for supplier {supplier} in cost matrix')
    m = gp.Model('FLP')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars([(i, j) for i in suppliers for j in supermarkets], lb=0, vtype=GRB.CONTINUOUS, name='')
    y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in suppliers for j in supermarkets)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in suppliers)) == demand[j] for j in supermarkets), name='')
    for i in suppliers:
        for j in supermarkets:
            m.addConstr(x_vars[i, j] <= demand[j] * y_vars[i], name=f'activation_{i}_{j}')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')