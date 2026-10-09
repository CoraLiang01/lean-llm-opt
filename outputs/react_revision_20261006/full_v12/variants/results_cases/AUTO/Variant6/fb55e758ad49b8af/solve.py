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
             'filters': {'conditions': [], 'logic': 'and'},
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
             'filters': {'conditions': [], 'logic': 'and'},
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

def solve_problem(CSVQA_FRAMES):
    option_df = CSVQA_FRAMES['file_0_view_0']
    limits_df = CSVQA_FRAMES['file_1_view_0']
    families = option_df['Family'].unique().tolist()
    options_per_family = {}
    for f in families:
        options_per_family[f] = option_df[option_df['Family'] == f]['Option'].unique().tolist()
    fo_keys = []
    for f in families:
        for o in options_per_family[f]:
            fo_keys.append((f, o))
    v_fo = {}
    w_fo = {}
    b_fo = {}
    for (_, row) in option_df.iterrows():
        f = row['Family']
        o = row['Option']
        v_fo[f, o] = float(row['Value'])
        w_fo[f, o] = float(row['Weight'])
        b_fo[f, o] = float(row['BudgetUse'])
    Wmax = None
    Bmax = None
    for (_, row) in limits_df.iterrows():
        resource = row['Resource']
        limit = float(row['Limit'])
        if resource == 'Weight':
            Wmax = limit
        elif resource == 'BudgetUse':
            Bmax = limit
    if Wmax is None or Bmax is None:
        raise ValueError('Missing resource limits for Weight or BudgetUse.')
    m = gp.Model('BinaryMultiChoiceKnapsack')
    x_vars = m.addVars(fo_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((v_fo[fo] * x_vars[fo] for fo in fo_keys)), gp.GRB.MAXIMIZE)
    for f in families:
        m.addConstr(gp.quicksum((x_vars[f, o] for o in options_per_family[f])) == 1, name=f'one_option_{f}')
    m.addConstr(gp.quicksum((w_fo[fo] * x_vars[fo] for fo in fo_keys)) <= Wmax, name='shelf_weight_limit')
    m.addConstr(gp.quicksum((b_fo[fo] * x_vars[fo] for fo in fo_keys)) <= Bmax, name='budget_limit')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)