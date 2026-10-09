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
             'role': 'supplier-supermarket transportation cost matrix',
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

def solve_problem():
    import pandas as pd
    I_df = CSVQA_FRAMES['file_1_view_0']
    J_df = CSVQA_FRAMES['file_0_view_0']
    C_df = CSVQA_FRAMES['file_2_view_0']
    I = list(I_df['Unnamed: 0'])
    J = list(J_df['customer'])
    f_i = {}
    for (idx, row) in I_df.iterrows():
        i = row['Unnamed: 0']
        try:
            f_i[i] = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Invalid fixed_costs for supplier {i}: {row['fixed_costs']}")
    d_j = {}
    for (idx, row) in J_df.iterrows():
        j = row['customer']
        try:
            d_j[j] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand for customer {j}: {row['demand']}")
    c_ij = {}
    for (idx, row) in C_df.iterrows():
        i = row['Unnamed: 0']
        if i not in I:
            raise ValueError(f'Supplier {i} in transportation_costs.csv not found in fixed_cost.csv')
        for j in J:
            if j not in row:
                raise ValueError(f'Supermarket {j} missing in transportation_costs.csv for supplier {i}')
            try:
                c_ij[i, j] = float(row[j])
            except Exception:
                raise ValueError(f'Invalid transportation cost for ({i},{j}): {row[j]}')
    for i in I:
        if i not in f_i:
            raise ValueError(f'Missing fixed cost for supplier {i}')
        for j in J:
            if (i, j) not in c_ij:
                raise ValueError(f'Missing transportation cost for ({i},{j})')
    for j in J:
        if j not in d_j:
            raise ValueError(f'Missing demand for customer {j}')
    m = gp.Model('UFLP')
    y_vars = m.addVars(I, vtype=GRB.BINARY, name='')
    x_vars = m.addVars([(i, j) for i in I for j in J], lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((f_i[i] * y_vars[i] for i in I)) + gp.quicksum((c_ij[i, j] * x_vars[i, j] for i in I for j in J)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in I)) == d_j[j], name=f'demand_{j}')
    for i in I:
        for j in J:
            m.addConstr(x_vars[i, j] <= d_j[j] * y_vars[i], name=f'link_{i}_{j}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print(m.Status)