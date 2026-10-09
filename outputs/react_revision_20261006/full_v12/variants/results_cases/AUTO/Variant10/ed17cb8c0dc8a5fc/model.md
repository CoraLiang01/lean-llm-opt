## Mathematical Model

Let:
- $V$ = set of nodes (from network_nodes.csv), indexed by $i$
- $E$ = set of undirected edges (from network_edges.csv), each $e = \{i,j\}$ with cost $c_e$
- $r$ = root node (from network_parameters.csv, $r = $ N1)
- For each undirected edge $e = \{i,j\} \in E$, define binary variable $y_{ij}$: $y_{ij} = 1$ if edge $\{i,j\}$ is built, $0$ otherwise
- For each directed arc $(i,j)$ where $\{i,j\} \in E$, define continuous variable $f_{ij} \geq 0$: flow from $i$ to $j$

**Objective:**
\[
\min \sum_{\{i,j\} \in E} c_{ij} \, y_{ij}
\]

**Subject to:**

1. **Spanning Tree Cardinality:**
\[
\sum_{\{i,j\} \in E} y_{ij} = |V| - 1
\]

2. **Flow Conservation (connectivity):**
\[
\sum_{j : \{r,j\} \in E} f_{rj} - \sum_{j : \{j,r\} \in E} f_{jr} = |V| - 1
\]
\[
\forall i \in V \setminus \{r\}:\quad
\sum_{j : \{i,j\} \in E} f_{ij} - \sum_{j : \{j,i\} \in E} f_{ji} = -1
\]

3. **Flow-Edge Linking:**
\[
\forall \{i,j\} \in E:\quad
f_{ij} \leq (|V| - 1) \, y_{ij}
\]
\[
f_{ji} \leq (|V| - 1) \, y_{ij}
\]

4. **Variable Domains:**
\[
y_{ij} \in \{0,1\} \quad \forall \{i,j\} \in E
\]
\[
f_{ij} \geq 0 \quad \forall (i,j): \{i,j\} \in E
\]

---

### Data Mapping

- $V$ = $\{$N1, N2, N3, N4, N5, N6, N7, N8$\}$ (from network_nodes.csv, table_id: file_0_view_0, column: Node)
- $E$ = $\{$(N1,N2), (N1,N3), (N1,N4), (N2,N3), (N2,N5), (N3,N4), (N3,N6), (N4,N6), (N4,N7), (N5,N6), (N5,N8), (N6,N7), (N6,N8), (N7,N8)$\}$ (from network_edges.csv, table_id: file_1_view_0, columns: Node1, Node2)
- $c_{ij}$ = ConstructionCost for edge $\{i,j\}$ (from network_edges.csv, table_id: file_1_view_0, column: ConstructionCost)
- $r$ = N1 (from network_parameters.csv, table_id: file_2_view_0, column: Value where Parameter = RootNode)

All indices, parameters, and constraints are defined exactly as above.