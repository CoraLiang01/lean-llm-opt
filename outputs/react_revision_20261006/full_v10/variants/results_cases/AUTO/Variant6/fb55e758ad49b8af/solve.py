CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retailer is designing a promotional bundle composed of one option from each product family. The value, '
          'shelf weight, and budget usage of each option are given in option_catalog.csv. The overall shelf-weight and '
          'budget limits are given in resource_limits.csv.\n'
          '\n'
          'Formulate a binary multi-choice knapsack model. For each family f and option o, define x_fo as a binary '
          'variable equal to 1 if option o is selected for family f and 0 otherwise. The objective is to maximize '
          'total bundle value. The model should select exactly one option from each family, satisfy the total weight '
          'and budget limits, and impose binary restrictions on all selection variables.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Family', 'Option', 'Value', 'Weight', 'BudgetUse'],
             'file_index': 0,
             'file_name': 'option_catalog.csv',
             'filters': {},
             'original_rows': 24,
             'records': [{'source_row': 0,
                          'values': {'BudgetUse': '10', 'Family': 'F1', 'Option': 'O1', 'Value': '28', 'Weight': '8'}},
                         {'source_row': 1,
                          'values': {'BudgetUse': '13', 'Family': 'F1', 'Option': 'O2', 'Value': '34', 'Weight': '11'}},
                         {'source_row': 2,
                          'values': {'BudgetUse': '12', 'Family': 'F1', 'Option': 'O3', 'Value': '30', 'Weight': '9'}},
                         {'source_row': 3,
                          'values': {'BudgetUse': '9', 'Family': 'F1', 'Option': 'O4', 'Value': '24', 'Weight': '7'}},
                         {'source_row': 4,
                          'values': {'BudgetUse': '9', 'Family': 'F2', 'Option': 'O1', 'Value': '25', 'Weight': '7'}},
                         {'source_row': 5,
                          'values': {'BudgetUse': '12', 'Family': 'F2', 'Option': 'O2', 'Value': '31', 'Weight': '10'}},
                         {'source_row': 6,
                          'values': {'BudgetUse': '14', 'Family': 'F2', 'Option': 'O3', 'Value': '36', 'Weight': '12'}},
                         {'source_row': 7,
                          'values': {'BudgetUse': '11', 'Family': 'F2', 'Option': 'O4', 'Value': '29', 'Weight': '9'}},
                         {'source_row': 8,
                          'values': {'BudgetUse': '12', 'Family': 'F3', 'Option': 'O1', 'Value': '33', 'Weight': '10'}},
                         {'source_row': 9,
                          'values': {'BudgetUse': '10', 'Family': 'F3', 'Option': 'O2', 'Value': '27', 'Weight': '8'}},
                         {'source_row': 10,
                          'values': {'BudgetUse': '15', 'Family': 'F3', 'Option': 'O3', 'Value': '38', 'Weight': '13'}},
                         {'source_row': 11,
                          'values': {'BudgetUse': '11', 'Family': 'F3', 'Option': 'O4', 'Value': '30', 'Weight': '9'}},
                         {'source_row': 12,
                          'values': {'BudgetUse': '8', 'Family': 'F4', 'Option': 'O1', 'Value': '26', 'Weight': '6'}},
                         {'source_row': 13,
                          'values': {'BudgetUse': '13', 'Family': 'F4', 'Option': 'O2', 'Value': '35', 'Weight': '11'}},
                         {'source_row': 14,
                          'values': {'BudgetUse': '12', 'Family': 'F4', 'Option': 'O3', 'Value': '32', 'Weight': '10'}},
                         {'source_row': 15,
                          'values': {'BudgetUse': '10', 'Family': 'F4', 'Option': 'O4', 'Value': '28', 'Weight': '8'}},
                         {'source_row': 16,
                          'values': {'BudgetUse': '11', 'Family': 'F5', 'Option': 'O1', 'Value': '30', 'Weight': '9'}},
                         {'source_row': 17,
                          'values': {'BudgetUse': '14', 'Family': 'F5', 'Option': 'O2', 'Value': '37', 'Weight': '12'}},
                         {'source_row': 18,
                          'values': {'BudgetUse': '10', 'Family': 'F5', 'Option': 'O3', 'Value': '29', 'Weight': '8'}},
                         {'source_row': 19,
                          'values': {'BudgetUse': '13', 'Family': 'F5', 'Option': 'O4', 'Value': '34', 'Weight': '11'}},
                         {'source_row': 20,
                          'values': {'BudgetUse': '9', 'Family': 'F6', 'Option': 'O1', 'Value': '24', 'Weight': '7'}},
                         {'source_row': 21,
                          'values': {'BudgetUse': '12', 'Family': 'F6', 'Option': 'O2', 'Value': '32', 'Weight': '10'}},
                         {'source_row': 22,
                          'values': {'BudgetUse': '14', 'Family': 'F6', 'Option': 'O3', 'Value': '36', 'Weight': '12'}},
                         {'source_row': 23,
                          'values': {'BudgetUse': '11', 'Family': 'F6', 'Option': 'O4', 'Value': '31', 'Weight': '9'}}],
             'returned_rows': 24,
             'role': 'option catalog',
             'table_id': 'file_0_view_0'},
            {'columns': ['Resource', 'Limit'],
             'file_index': 1,
             'file_name': 'resource_limits.csv',
             'filters': {},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'Limit': '55', 'Resource': 'Weight'}},
                         {'source_row': 1, 'values': {'Limit': '70', 'Resource': 'BudgetUse'}}],
             'returned_rows': 2,
             'role': 'resource limits',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd
import numpy as np
import sys

def solve_problem(CSVQA_FRAMES):
    option_frame = CSVQA_FRAMES['file_0_view_0']
    resource_frame = CSVQA_FRAMES['file_1_view_0']
    families = list(option_frame['Family'].unique())
    family_options = {}
    for f in families:
        family_options[f] = list(option_frame[option_frame['Family'] == f]['Option'].unique())
    v_fo = {}
    w_fo = {}
    b_fo = {}
    option_keys = []
    for (idx, row) in option_frame.iterrows():
        f = row['Family']
        o = row['Option']
        key = (f, o)
        option_keys.append(key)
        try:
            v_fo[key] = float(row['Value'])
            w_fo[key] = float(row['Weight'])
            b_fo[key] = float(row['BudgetUse'])
        except Exception as e:
            raise ValueError(f'Non-numeric value in Value/Weight/BudgetUse for Family={f}, Option={o}: {e}')
    L_r = {}
    for (idx, row) in resource_frame.iterrows():
        r = row['Resource']
        try:
            L_r[r] = float(row['Limit'])
        except Exception as e:
            raise ValueError(f'Non-numeric value in Limit for Resource={r}: {e}')
    if 'Weight' not in L_r or 'BudgetUse' not in L_r:
        raise KeyError("Resource limits must include both 'Weight' and 'BudgetUse'.")
    for f in families:
        if f not in family_options or len(family_options[f]) == 0:
            raise ValueError(f'Family {f} has no options.')
        for o in family_options[f]:
            if (f, o) not in v_fo or (f, o) not in w_fo or (f, o) not in b_fo:
                raise ValueError(f'Missing data for Family={f}, Option={o}.')
    m = gp.Model('MultiChoiceKnapsack')
    x_vars = m.addVars(option_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((v_fo[key] * x_vars[key] for key in option_keys)), gp.GRB.MAXIMIZE)
    for f in families:
        m.addConstr(gp.quicksum((x_vars[f, o] for o in family_options[f])) == 1, name=f'OneOption_{f}')
    m.addConstr(gp.quicksum((w_fo[key] * x_vars[key] for key in option_keys)) <= L_r['Weight'], name='WeightLimit')
    m.addConstr(gp.quicksum((b_fo[key] * x_vars[key] for key in option_keys)) <= L_r['BudgetUse'], name='BudgetUseLimit')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for key in option_keys:
            print(f'{x_vars[key].VarName} {x_vars[key].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem(CSVQA_FRAMES)