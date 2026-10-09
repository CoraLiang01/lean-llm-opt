CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the construction industry, we have a set of managers and a set of construction projects. Each manager '
          'incurs different costs for each project, based on their experience and expertise. The cost information for '
          'each manager and project is saved in the CSV file "manager_project_costs.csv", where each row represents a '
          'manager, and each column represents the cost for that manager to complete a specific project. The objective '
          'is to find the optimal assignment that minimizes the total cost of completing all projects. Each manager '
          'must be assigned to exactly one project, and each project must be managed by exactly one manager. The goal '
          'is to minimize the total cost while satisfying these one-to-one assignment constraints.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'columns': ['Unnamed: 0', 'P1', 'P2', 'P3'],
             'file_index': 0,
             'file_name': 'manager_project_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'P1': '3000', 'P2': '3200', 'P3': '3100', 'Unnamed: 0': 'MA'}},
                         {'source_row': 1, 'values': {'P1': '2800', 'P2': '3300', 'P3': '2900', 'Unnamed: 0': 'MB'}},
                         {'source_row': 2, 'values': {'P1': '2900', 'P2': '3100', 'P3': '3000', 'Unnamed: 0': 'MC'}}],
             'returned_rows': 3,
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_0_view_0', 'shape': [3, 3], "
                                   "'expected_shape': [3, 3], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_0_view_0', 'shape': [3, 3], "
                                   "'expected_shape': [3, 3], 'row_ids_aligned': True, 'column_ids_aligned': False, "
                                   "'row_mapping_basis': 'exact', 'column_mapping_basis': 'unresolved'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def solve_problem():
    df = CSVQA_FRAMES['file_0_view_0']
    M = list(df['Unnamed: 0'])
    P = ['P1', 'P2', 'P3']
    c_mp = {}
    for (idx, row) in df.iterrows():
        m = row['Unnamed: 0']
        c_mp[m] = {}
        for p in P:
            val = row[p]
            try:
                c_mp[m][p] = float(val)
            except Exception:
                raise ValueError(f'Non-numeric or missing cost for manager {m}, project {p}: {val}')
    if set(c_mp.keys()) != set(M):
        raise ValueError('Mismatch in manager keys between data and set M')
    for m in M:
        if set(c_mp[m].keys()) != set(P):
            raise ValueError(f'Mismatch in project keys for manager {m}')
    m_model = gp.Model()
    m_model.Params.MIPGap = 0.0001
    x_keys = [(m, p) for m in M for p in P]
    x_vars = m_model.addVars(x_keys, vtype=GRB.BINARY, name='')
    m_model.setObjective(gp.quicksum((c_mp[m][p] * x_vars[m, p] for m in M for p in P)), GRB.MINIMIZE)
    for m in M:
        m_model.addConstr(gp.quicksum((x_vars[m, p] for p in P)) == 1, name='assign_mgr_' + m)
    for p in P:
        m_model.addConstr(gp.quicksum((x_vars[m, p] for m in M)) == 1, name='assign_proj_' + p)
    m_model.optimize()
    if m_model.Status == GRB.OPTIMAL:
        print(m_model.ObjVal)
        for key in x_keys:
            var = x_vars[key]
            print(var.VarName, var.X)
    else:
        print(m_model.Status)
    return m_model
m = solve_problem()