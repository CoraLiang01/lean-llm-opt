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
                                         'inclusive': 'both',
                                         'operator': 'exact',
                                         'value': 'RootNode'}],
                         'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Parameter': 'RootNode', 'Value': 'V1'}}],
             'returned_rows': 1,
             'role': 'root node parameter',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    nodes_df = CSVQA_FRAMES['file_0_view_0']
    nodes = list(nodes_df['Node'])
    n = len(nodes)
    if n == 0:
        raise ValueError('No nodes found in network_nodes.csv')
    node_set = set(nodes)
    edges_df = CSVQA_FRAMES['file_1_view_0']
    edge_keys = []
    edge_cost = {}
    for (idx, row) in edges_df.iterrows():
        i = row['Node1']
        j = row['Node2']
        if i not in node_set or j not in node_set:
            raise ValueError(f'Edge ({i},{j}) contains node not in node set')
        if i < j:
            key = (i, j)
        else:
            key = (j, i)
        if key in edge_cost:
            raise ValueError(f'Duplicate edge ({i},{j}) in network_edges.csv')
        edge_keys.append(key)
        try:
            cost = float(row['ConstructionCost'])
        except Exception:
            raise ValueError(f'Invalid ConstructionCost for edge ({i},{j})')
        edge_cost[key] = cost
    E = edge_keys
    if len(E) == 0:
        raise ValueError('No edges found in network_edges.csv')
    arc_keys = []
    for (i, j) in E:
        arc_keys.append((i, j))
        arc_keys.append((j, i))
    A = arc_keys
    param_df = CSVQA_FRAMES['file_2_view_0']
    root_row = param_df[param_df['Parameter'].str.casefold() == 'rootnode']
    if root_row.empty:
        raise ValueError('RootNode parameter not found in network_parameters.csv')
    r = root_row.iloc[0]['Value']
    if r not in node_set:
        raise ValueError(f'Root node {r} not in node set')
    m = gp.Model('MinimumCostSpanningTree')
    y_vars = m.addVars(E, vtype=gp.GRB.BINARY, name='')
    f_vars = m.addVars(A, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((edge_cost[e] * y_vars[e] for e in E)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((y_vars[e] for e in E)) == n - 1, name='EdgeCount')
    m.addConstr(gp.quicksum((f_vars[r, j] for j in nodes if (r, j) in f_vars)) - gp.quicksum((f_vars[j, r] for j in nodes if (j, r) in f_vars)) == n - 1, name='RootFlowBalance')
    for i in nodes:
        if i == r:
            continue
        m.addConstr(gp.quicksum((f_vars[i, j] for j in nodes if (i, j) in f_vars)) - gp.quicksum((f_vars[j, i] for j in nodes if (j, i) in f_vars)) == -1, name=f'FlowBalance_{i}')
    M = n - 1
    for (i, j) in E:
        m.addConstr(f_vars[i, j] <= M * y_vars[i, j], name=f'FlowLink_{i}_{j}_fij')
        m.addConstr(f_vars[j, i] <= M * y_vars[i, j], name=f'FlowLink_{i}_{j}_fji')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')