CSVQA_DATA = {'query': 'A product team must select exactly one design option from each component family. Option value, weight, and budget use are listed in option_catalog.csv, and total resource limits are listed in resource_limits.csv.\n\nFormulate a maximum-value multi-choice knapsack model. For each family-option pair g-o, define x_go as a binary variable equal to 1 if option o is selected from family g. The objective is to maximize total selected value. The model should include exactly-one-option constraints for every family, total weight and budget-use constraints, and binary restrictions for all option-selection variables.', 'relationships': [], 'tables': [{'columns': ['Family', 'Option', 'Value', 'Weight', 'BudgetUse'], 'file_name': 'option_catalog.csv', 'records': [{'data_row_number': 0, 'values': {'BudgetUse': '8', 'Family': 'G1', 'Option': 'O1', 'Value': '22', 'Weight': '6'}}, {'data_row_number': 1, 'values': {'BudgetUse': '11', 'Family': 'G1', 'Option': 'O2', 'Value': '29', 'Weight': '9'}}, {'data_row_number': 2, 'values': {'BudgetUse': '13', 'Family': 'G1', 'Option': 'O3', 'Value': '31', 'Weight': '10'}}, {'data_row_number': 3, 'values': {'BudgetUse': '9', 'Family': 'G1', 'Option': 'O4', 'Value': '25', 'Weight': '7'}}, {'data_row_number': 4, 'values': {'BudgetUse': '8', 'Family': 'G2', 'Option': 'O1', 'Value': '24', 'Weight': '7'}}, {'data_row_number': 5, 'values': {'BudgetUse': '13', 'Family': 'G2', 'Option': 'O2', 'Value': '33', 'Weight': '11'}}, {'data_row_number': 6, 'values': {'BudgetUse': '10', 'Family': 'G2', 'Option': 'O3', 'Value': '28', 'Weight': '8'}}, {'data_row_number': 7, 'values': {'BudgetUse': '14', 'Family': 'G2', 'Option': 'O4', 'Value': '35', 'Weight': '12'}}, {'data_row_number': 8, 'values': {'BudgetUse': '12', 'Family': 'G3', 'Option': 'O1', 'Value': '30', 'Weight': '9'}}, {'data_row_number': 9, 'values': {'BudgetUse': '9', 'Family': 'G3', 'Option': 'O2', 'Value': '26', 'Weight': '7'}}, {'data_row_number': 10, 'values': {'BudgetUse': '16', 'Family': 'G3', 'Option': 'O3', 'Value': '38', 'Weight': '13'}}, {'data_row_number': 11, 'values': {'BudgetUse': '13', 'Family': 'G3', 'Option': 'O4', 'Value': '34', 'Weight': '11'}}, {'data_row_number': 12, 'values': {'BudgetUse': '7', 'Family': 'G4', 'Option': 'O1', 'Value': '21', 'Weight': '5'}}, {'data_row_number': 13, 'values': {'BudgetUse': '12', 'Family': 'G4', 'Option': 'O2', 'Value': '32', 'Weight': '10'}}, {'data_row_number': 14, 'values': {'BudgetUse': '15', 'Family': 'G4', 'Option': 'O3', 'Value': '36', 'Weight': '12'}}, {'data_row_number': 15, 'values': {'BudgetUse': '10', 'Family': 'G4', 'Option': 'O4', 'Value': '27', 'Weight': '8'}}, {'data_row_number': 16, 'values': {'BudgetUse': '11', 'Family': 'G5', 'Option': 'O1', 'Value': '29', 'Weight': '8'}}, {'data_row_number': 17, 'values': {'BudgetUse': '15', 'Family': 'G5', 'Option': 'O2', 'Value': '37', 'Weight': '12'}}, {'data_row_number': 18, 'values': {'BudgetUse': '12', 'Family': 'G5', 'Option': 'O3', 'Value': '33', 'Weight': '10'}}, {'data_row_number': 19, 'values': {'BudgetUse': '8', 'Family': 'G5', 'Option': 'O4', 'Value': '24', 'Weight': '6'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Resource', 'Limit'], 'file_name': 'resource_limits.csv', 'records': [{'data_row_number': 0, 'values': {'Limit': '48', 'Resource': 'Weight'}}, {'data_row_number': 1, 'values': {'Limit': '60', 'Resource': 'BudgetUse'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    option_table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if option_table is None:
        raise ValueError("Option catalog table 'file_0_view_0' not found")
    limit_table = next((t for t in tables if t.get('table_id') == 'file_1_view_0'), None)
    if limit_table is None:
        raise ValueError("Resource limits table 'file_1_view_0' not found")
    records = option_table['records']
    families = set()
    options = set()
    v = {}
    w = {}
    b = {}
    for rec in records:
        vals = rec['values']
        g = vals['Family']
        o = vals['Option']
        families.add(g)
        options.add(o)
        try:
            v[g, o] = int(vals['Value'])
            w[g, o] = int(vals['Weight'])
            b[g, o] = int(vals['BudgetUse'])
        except (KeyError, ValueError) as e:
            raise ValueError(f"Missing or invalid parameter in record {rec['data_row_number']}: {e}")
    limits = {}
    for rec in limit_table['records']:
        vals = rec['values']
        res = vals['Resource']
        try:
            limits[res] = int(vals['Limit'])
        except (KeyError, ValueError) as e:
            raise ValueError(f'Missing or invalid limit for resource {res}: {e}')
    if 'Weight' not in limits or 'BudgetUse' not in limits:
        raise ValueError('Required resource limits not found')
    Wmax = limits['Weight']
    Bmax = limits['BudgetUse']
    for g in families:
        if not any(((g, o) in v for o in options)):
            raise ValueError(f'Family {g} has no options')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    keys = [(g, o) for g in families for o in options if (g, o) in v]
    x = m.addVars(keys, vtype=GRB.BINARY, name='')
    obj = gurobipy.quicksum((v[g, o] * x[g, o] for g, o in keys))
    m.setObjective(obj, GRB.MAXIMIZE)
    for g in families:
        m.addConstr(gurobipy.quicksum((x[g, o] for o in options if (g, o) in x)) == 1, name=f'one_option_{g}')
    m.addConstr(gurobipy.quicksum((w[g, o] * x[g, o] for g, o in keys)) <= Wmax, name='weight_capacity')
    m.addConstr(gurobipy.quicksum((b[g, o] * x[g, o] for g, o in keys)) <= Bmax, name='budget_capacity')
    m.optimize()
    return m
m = solve_problem()
