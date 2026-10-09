CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A utility company must build a least-cost communication backbone connecting all substations in a region. '
          'The complete node set is listed in network_nodes.csv, the available undirected links and their construction '
          'costs are given in network_edges.csv, and the root node used for the flow-based connectivity formulation is '
          'given in network_parameters.csv.\n'
          '\n'
          'Formulate a minimum spanning tree network-design model. For each candidate link e, define y_e as a binary '
          'variable equal to 1 if link e is built and 0 otherwise. To enforce connectivity, define directed auxiliary '
          'flow variables f_ij on both directions of each candidate link, sending one unit of flow from the root to '
          'every other node. The objective is to minimize total construction cost. The model should select exactly n-1 '
          'links, where n is the number of nodes in network_nodes.csv, satisfy the flow-conservation connectivity '
          'constraints, link auxiliary flow to selected links, impose nonnegativity on flow variables, and impose '
          'binary restrictions on link-selection variables.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Node'],
             'file_index': 0,
             'file_name': 'network_nodes.csv',
             'filters': {},
             'original_rows': 8,
             'records': [{'source_row': 0, 'values': {'Node': 'N1'}},
                         {'source_row': 1, 'values': {'Node': 'N2'}},
                         {'source_row': 2, 'values': {'Node': 'N3'}},
                         {'source_row': 3, 'values': {'Node': 'N4'}},
                         {'source_row': 4, 'values': {'Node': 'N5'}},
                         {'source_row': 5, 'values': {'Node': 'N6'}},
                         {'source_row': 6, 'values': {'Node': 'N7'}},
                         {'source_row': 7, 'values': {'Node': 'N8'}}],
             'returned_rows': 8,
             'role': 'node list',
             'table_id': 'file_0_view_0'},
            {'columns': ['Node1', 'Node2', 'ConstructionCost'],
             'file_index': 1,
             'file_name': 'network_edges.csv',
             'filters': {},
             'original_rows': 14,
             'records': [{'source_row': 0, 'values': {'ConstructionCost': '4', 'Node1': 'N1', 'Node2': 'N2'}},
                         {'source_row': 1, 'values': {'ConstructionCost': '3', 'Node1': 'N1', 'Node2': 'N3'}},
                         {'source_row': 2, 'values': {'ConstructionCost': '9', 'Node1': 'N1', 'Node2': 'N4'}},
                         {'source_row': 3, 'values': {'ConstructionCost': '5', 'Node1': 'N2', 'Node2': 'N3'}},
                         {'source_row': 4, 'values': {'ConstructionCost': '6', 'Node1': 'N2', 'Node2': 'N5'}},
                         {'source_row': 5, 'values': {'ConstructionCost': '4', 'Node1': 'N3', 'Node2': 'N4'}},
                         {'source_row': 6, 'values': {'ConstructionCost': '7', 'Node1': 'N3', 'Node2': 'N6'}},
                         {'source_row': 7, 'values': {'ConstructionCost': '2', 'Node1': 'N4', 'Node2': 'N6'}},
                         {'source_row': 8, 'values': {'ConstructionCost': '8', 'Node1': 'N4', 'Node2': 'N7'}},
                         {'source_row': 9, 'values': {'ConstructionCost': '3', 'Node1': 'N5', 'Node2': 'N6'}},
                         {'source_row': 10, 'values': {'ConstructionCost': '10', 'Node1': 'N5', 'Node2': 'N8'}},
                         {'source_row': 11, 'values': {'ConstructionCost': '4', 'Node1': 'N6', 'Node2': 'N7'}},
                         {'source_row': 12, 'values': {'ConstructionCost': '6', 'Node1': 'N6', 'Node2': 'N8'}},
                         {'source_row': 13, 'values': {'ConstructionCost': '5', 'Node1': 'N7', 'Node2': 'N8'}}],
             'returned_rows': 14,
             'role': 'edge list with costs',
             'table_id': 'file_1_view_0'},
            {'columns': ['Parameter', 'Value'],
             'file_index': 2,
             'file_name': 'network_parameters.csv',
             'filters': {'conditions': [{'column': 'Parameter',
                                         'dtype': 'string',
                                         'evidence': 'the root node used for the flow-based connectivity formulation '
                                                     'is given in network_parameters.csv',
                                         'operator': 'exact',
                                         'value': 'RootNode'}],
                         'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Parameter': 'RootNode', 'Value': 'N1'}}],
             'returned_rows': 1,
             'role': 'root node parameter',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd

def solve_problem():
    nodes_df = CSVQA_FRAMES['file_0_view_0']
    edges_df = CSVQA_FRAMES['file_1_view_0']
    params_df = CSVQA_FRAMES['file_2_view_0']
    N = []
    for (_, row) in nodes_df.iterrows():
        node = row['Node']
        N.append(node)
    n_nodes = len(N)
    E = []
    c = {}
    for (_, row) in edges_df.iterrows():
        i = row['Node1']
        j = row['Node2']
        cost = float(row['ConstructionCost'])
        edge = tuple(sorted((i, j)))
        E.append(edge)
        c[edge] = cost
    A = []
    edge_to_arcs = {}
    for (i, j) in E:
        A.append((i, j))
        A.append((j, i))
        edge_to_arcs[i, j] = [(i, j), (j, i)]
        edge_to_arcs[j, i] = [(i, j), (j, i)]
    root_row = params_df[params_df['Parameter'].str.casefold() == 'rootnode']
    if root_row.empty:
        raise ValueError('RootNode parameter not found in network_parameters.csv')
    r = root_row.iloc[0]['Value']
    m = gp.Model('MST_Network_Design')
    y_vars = m.addVars(E, vtype=gp.GRB.BINARY, name='')
    f_vars = m.addVars(A, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((c[edge] * y_vars[edge] for edge in E)), gp.GRB.MINIMIZE)
    m.addConstr(y_vars.sum() == n_nodes - 1, name='link_count')
    for k in N:
        inflow = gp.quicksum((f_vars[i, k] for (i, j) in A if j == k))
        outflow = gp.quicksum((f_vars[k, j] for (i, j) in A if i == k))
        if k == r:
            m.addConstr(inflow - outflow == 1 - (n_nodes - 1), name=f'flow_conserv_{k}')
        else:
            m.addConstr(inflow - outflow == 1, name=f'flow_conserv_{k}')
    for (u, v) in A:
        edge = tuple(sorted((u, v)))
        if edge in E:
            m.addConstr(f_vars[u, v] <= (n_nodes - 1) * y_vars[edge], name=f'flow_link_{u}_{v}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()