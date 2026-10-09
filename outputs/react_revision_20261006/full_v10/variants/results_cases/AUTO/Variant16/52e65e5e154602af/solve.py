CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A product team must select exactly one design option from each component family. Option value, weight, and '
          'budget use are listed in option_catalog.csv, and total resource limits are listed in resource_limits.csv.\n'
          '\n'
          'Formulate a maximum-value multi-choice knapsack model. For each family-option pair g-o, define x_go as a '
          'binary variable equal to 1 if option o is selected from family g. The objective is to maximize total '
          'selected value. The model should include exactly-one-option constraints for every family, total weight and '
          'budget-use constraints, and binary restrictions for all option-selection variables.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Family', 'Option', 'Value', 'Weight', 'BudgetUse'],
             'file_index': 0,
             'file_name': 'option_catalog.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0,
                          'values': {'BudgetUse': '8', 'Family': 'G1', 'Option': 'O1', 'Value': '22', 'Weight': '6'}},
                         {'source_row': 1,
                          'values': {'BudgetUse': '11', 'Family': 'G1', 'Option': 'O2', 'Value': '29', 'Weight': '9'}},
                         {'source_row': 2,
                          'values': {'BudgetUse': '13', 'Family': 'G1', 'Option': 'O3', 'Value': '31', 'Weight': '10'}},
                         {'source_row': 3,
                          'values': {'BudgetUse': '9', 'Family': 'G1', 'Option': 'O4', 'Value': '25', 'Weight': '7'}},
                         {'source_row': 4,
                          'values': {'BudgetUse': '8', 'Family': 'G2', 'Option': 'O1', 'Value': '24', 'Weight': '7'}},
                         {'source_row': 5,
                          'values': {'BudgetUse': '13', 'Family': 'G2', 'Option': 'O2', 'Value': '33', 'Weight': '11'}},
                         {'source_row': 6,
                          'values': {'BudgetUse': '10', 'Family': 'G2', 'Option': 'O3', 'Value': '28', 'Weight': '8'}},
                         {'source_row': 7,
                          'values': {'BudgetUse': '14', 'Family': 'G2', 'Option': 'O4', 'Value': '35', 'Weight': '12'}},
                         {'source_row': 8,
                          'values': {'BudgetUse': '12', 'Family': 'G3', 'Option': 'O1', 'Value': '30', 'Weight': '9'}},
                         {'source_row': 9,
                          'values': {'BudgetUse': '9', 'Family': 'G3', 'Option': 'O2', 'Value': '26', 'Weight': '7'}},
                         {'source_row': 10,
                          'values': {'BudgetUse': '16', 'Family': 'G3', 'Option': 'O3', 'Value': '38', 'Weight': '13'}},
                         {'source_row': 11,
                          'values': {'BudgetUse': '13', 'Family': 'G3', 'Option': 'O4', 'Value': '34', 'Weight': '11'}},
                         {'source_row': 12,
                          'values': {'BudgetUse': '7', 'Family': 'G4', 'Option': 'O1', 'Value': '21', 'Weight': '5'}},
                         {'source_row': 13,
                          'values': {'BudgetUse': '12', 'Family': 'G4', 'Option': 'O2', 'Value': '32', 'Weight': '10'}},
                         {'source_row': 14,
                          'values': {'BudgetUse': '15', 'Family': 'G4', 'Option': 'O3', 'Value': '36', 'Weight': '12'}},
                         {'source_row': 15,
                          'values': {'BudgetUse': '10', 'Family': 'G4', 'Option': 'O4', 'Value': '27', 'Weight': '8'}},
                         {'source_row': 16,
                          'values': {'BudgetUse': '11', 'Family': 'G5', 'Option': 'O1', 'Value': '29', 'Weight': '8'}},
                         {'source_row': 17,
                          'values': {'BudgetUse': '15', 'Family': 'G5', 'Option': 'O2', 'Value': '37', 'Weight': '12'}},
                         {'source_row': 18,
                          'values': {'BudgetUse': '12', 'Family': 'G5', 'Option': 'O3', 'Value': '33', 'Weight': '10'}},
                         {'source_row': 19,
                          'values': {'BudgetUse': '8', 'Family': 'G5', 'Option': 'O4', 'Value': '24', 'Weight': '6'}}],
             'returned_rows': 20,
             'role': 'option catalog',
             'table_id': 'file_0_view_0'},
            {'columns': ['Resource', 'Limit'],
             'file_index': 1,
             'file_name': 'resource_limits.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'Limit': '48', 'Resource': 'Weight'}},
                         {'source_row': 1, 'values': {'Limit': '60', 'Resource': 'BudgetUse'}}],
             'returned_rows': 2,
             'role': 'resource limits',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    option_frame = CSVQA_FRAMES['file_0_view_0']
    limits_frame = CSVQA_FRAMES['file_1_view_0']
    families = option_frame['Family'].unique().tolist()
    options_per_family = {}
    for fam in families:
        options_per_family[fam] = option_frame[option_frame['Family'] == fam]['Option'].unique().tolist()
    go_keys = []
    for fam in families:
        for opt in options_per_family[fam]:
            go_keys.append((fam, opt))
    v_go = {}
    w_go = {}
    b_go = {}
    for (idx, row) in option_frame.iterrows():
        fam = row['Family']
        opt = row['Option']
        v_go[fam, opt] = float(row['Value'])
        w_go[fam, opt] = float(row['Weight'])
        b_go[fam, opt] = float(row['BudgetUse'])
    W_max = None
    B_max = None
    for (idx, row) in limits_frame.iterrows():
        resource = row['Resource']
        limit = float(row['Limit'])
        if resource == 'Weight':
            W_max = limit
        elif resource == 'BudgetUse':
            B_max = limit
    if W_max is None or B_max is None:
        raise ValueError('Missing resource limits for Weight or BudgetUse.')
    for fam in families:
        if fam not in options_per_family or len(options_per_family[fam]) == 0:
            raise ValueError(f'Family {fam} has no options.')
        for opt in options_per_family[fam]:
            if (fam, opt) not in v_go or (fam, opt) not in w_go or (fam, opt) not in b_go:
                raise ValueError(f'Missing data for ({fam}, {opt}).')
    m = gp.Model('multi_choice_knapsack')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(go_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((v_go[key] * x_vars[key] for key in go_keys)), gp.GRB.MAXIMIZE)
    for fam in families:
        m.addConstr(gp.quicksum((x_vars[fam, opt] for opt in options_per_family[fam])) == 1, name=f'one_option_{fam}')
    m.addConstr(gp.quicksum((w_go[key] * x_vars[key] for key in go_keys)) <= W_max, name='weight_limit')
    m.addConstr(gp.quicksum((b_go[key] * x_vars[key] for key in go_keys)) <= B_max, name='budget_limit')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)