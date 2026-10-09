CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A communications team must build a connected backbone over all listed sites. The node set is listed in '
          'network_nodes.csv, available undirected links and construction costs are listed in network_edges.csv, and '
          'the root node for a single-commodity flow formulation is listed in network_parameters.csv.\n'
          '\n'
          'Formulate a minimum-cost spanning tree model. For each undirected edge i-j, define y_ij as a binary '
          'variable equal to 1 if the edge is selected. For each directed arc version of an available edge, define '
          'f_ij as the nonnegative connectivity flow sent from the root. The objective is to minimize selected edge '
          'construction cost. The model should include a constraint selecting exactly n-1 edges, root and non-root '
          'flow-balance constraints using the node set in network_nodes.csv, flow-to-edge linking constraints with M = '
          'n-1, nonnegativity constraints for flow variables, and binary restrictions for edge-selection variables.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Node'],
             'file_index': 0,
             'file_name': 'network_nodes.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 9,
             'records': [{'source_row': 0, 'values': {'Node': 'V1'}},
                         {'source_row': 1, 'values': {'Node': 'V2'}},
                         {'source_row': 2, 'values': {'Node': 'V3'}},
                         {'source_row': 3, 'values': {'Node': 'V4'}},
                         {'source_row': 4, 'values': {'Node': 'V5'}},
                         {'source_row': 5, 'values': {'Node': 'V6'}},
                         {'source_row': 6, 'values': {'Node': 'V7'}},
                         {'source_row': 7, 'values': {'Node': 'V8'}},
                         {'source_row': 8, 'values': {'Node': 'V9'}}],
             'returned_rows': 9,
             'role': 'node set',
             'table_id': 'file_0_view_0'},
            {'columns': ['Node1', 'Node2', 'ConstructionCost'],
             'file_index': 1,
             'file_name': 'network_edges.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 16,
             'records': [{'source_row': 0, 'values': {'ConstructionCost': '6', 'Node1': 'V1', 'Node2': 'V2'}},
                         {'source_row': 1, 'values': {'ConstructionCost': '4', 'Node1': 'V1', 'Node2': 'V3'}},
                         {'source_row': 2, 'values': {'ConstructionCost': '9', 'Node1': 'V1', 'Node2': 'V4'}},
                         {'source_row': 3, 'values': {'ConstructionCost': '5', 'Node1': 'V2', 'Node2': 'V3'}},
                         {'source_row': 4, 'values': {'ConstructionCost': '7', 'Node1': 'V2', 'Node2': 'V5'}},
                         {'source_row': 5, 'values': {'ConstructionCost': '3', 'Node1': 'V3', 'Node2': 'V4'}},
                         {'source_row': 6, 'values': {'ConstructionCost': '8', 'Node1': 'V3', 'Node2': 'V6'}},
                         {'source_row': 7, 'values': {'ConstructionCost': '4', 'Node1': 'V4', 'Node2': 'V6'}},
                         {'source_row': 8, 'values': {'ConstructionCost': '10', 'Node1': 'V4', 'Node2': 'V7'}},
                         {'source_row': 9, 'values': {'ConstructionCost': '6', 'Node1': 'V5', 'Node2': 'V6'}},
                         {'source_row': 10, 'values': {'ConstructionCost': '11', 'Node1': 'V5', 'Node2': 'V8'}},
                         {'source_row': 11, 'values': {'ConstructionCost': '5', 'Node1': 'V6', 'Node2': 'V7'}},
                         {'source_row': 12, 'values': {'ConstructionCost': '7', 'Node1': 'V6', 'Node2': 'V8'}},
                         {'source_row': 13, 'values': {'ConstructionCost': '4', 'Node1': 'V7', 'Node2': 'V8'}},
                         {'source_row': 14, 'values': {'ConstructionCost': '6', 'Node1': 'V7', 'Node2': 'V9'}},
                         {'source_row': 15, 'values': {'ConstructionCost': '5', 'Node1': 'V8', 'Node2': 'V9'}}],
             'returned_rows': 16,
             'role': 'edge set with costs',
             'table_id': 'file_1_view_0'},
            {'columns': ['Parameter', 'Value'],
             'file_index': 2,
             'file_name': 'network_parameters.csv',
             'filters': {'conditions': [{'column': 'Parameter',
                                         'dtype': 'string',
                                         'evidence': 'the root node for a single-commodity flow formulation is listed '
                                                     'in network_parameters.csv',
                                         'operator': 'exact',
                                         'value': 'RootNode'}],
                         'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Parameter': 'RootNode', 'Value': 'V1'}}],
             'returned_rows': 1,
             'role': 'model parameters',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    nodes_df = CSVQA_FRAMES['file_0_view_0']
    edges_df = CSVQA_FRAMES['file_1_view_0']
    params_df = CSVQA_FRAMES['file_2_view_0']
    N = []
    for (_, row) in nodes_df.iterrows():
        node = row['Node']
        if node in N:
            raise ValueError(f'Duplicate node: {node}')
        N.append(node)
    n = len(N)
    if n < 2:
        raise ValueError('At least two nodes are required.')
    E = []
    c = {}
    for (_, row) in edges_df.iterrows():
        i = row['Node1']
        j = row['Node2']
        if i == j:
            raise ValueError(f'Self-loop edge: {i}-{j}')
        (k, l) = sorted([i, j])
        key = (k, l)
        if key in c:
            raise ValueError(f'Duplicate undirected edge: {k}-{l}')
        try:
            cost = float(row['ConstructionCost'])
        except Exception:
            raise ValueError(f"Invalid ConstructionCost for edge {k}-{l}: {row['ConstructionCost']}")
        E.append(key)
        c[key] = cost
    A = []
    for (i, j) in E:
        A.append((i, j))
        A.append((j, i))
    root_row = None
    for (_, row) in params_df.iterrows():
        if row['Parameter'].casefold() == 'rootnode':
            root_row = row
            break
    if root_row is None:
        raise ValueError('RootNode parameter not found in network_parameters.csv')
    r = root_row['Value']
    if r not in N:
        raise ValueError(f'Root node {r} not in node set.')
    M = n - 1
    for (i, j) in E:
        if i not in N or j not in N:
            raise ValueError(f'Edge ({i},{j}) contains node not in node set.')
    m = gp.Model('MinimumCostSpanningTree')
    y_vars = m.addVars(E, vtype=gp.GRB.BINARY, name='')
    f_vars = m.addVars(A, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((c[i, j] * y_vars[i, j] for (i, j) in E)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((y_vars[i, j] for (i, j) in E)) == n - 1, name='EdgeCount')
    m.addConstr(gp.quicksum((f_vars[r, j] for j in N if (r, j) in A)) - gp.quicksum((f_vars[j, r] for j in N if (j, r) in A)) == n - 1, name='RootFlowBalance')
    for i in N:
        if i == r:
            continue
        m.addConstr(gp.quicksum((f_vars[i, j] for j in N if (i, j) in A)) - gp.quicksum((f_vars[j, i] for j in N if (j, i) in A)) == -1, name=f'FlowBalance_{i}')
    for (i, j) in A:
        (k, l) = sorted([i, j])
        m.addConstr(f_vars[i, j] <= M * y_vars[k, l], name=f'FlowToEdge_{i}_{j}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')