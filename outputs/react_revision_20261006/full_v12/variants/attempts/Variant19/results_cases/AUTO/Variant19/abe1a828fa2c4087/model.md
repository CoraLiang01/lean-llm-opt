## Minimum-Cost Spanning Tree Model

**Sets**
- $N$: set of nodes (from network_nodes.csv), indexed by $i, j$
- $E$: set of undirected edges (from network_edges.csv), each edge $e = \{i,j\}$ with $i < j$
- $A$: set of directed arcs, i.e., for each $\{i,j\} \in E$, both $(i,j)$ and $(j,i)$
- $r$: root node (from network_parameters.csv, $r = $ V1)

**Parameters**
- $c_{ij}$: construction cost for undirected edge $\{i,j\} \in E$ (from network_edges.csv)
- $n$: $|N|$ (number of nodes)
- $M = n-1$

**Variables**
- $y_{ij} \in \{0,1\}$: 1 if undirected edge $\{i,j\}$ is selected, 0 otherwise, for all $\{i,j\} \in E$
- $f_{ij} \geq 0$: flow on directed arc $(i,j)$, for all $(i,j) \in A$

**Objective**
\[
\min \sum_{\{i,j\} \in E} c_{ij} \, y_{ij}
\]

**Constraints**

1. **Edge count (spanning tree):**
\[
\sum_{\{i,j\} \in E} y_{ij} = n-1
\]

2. **Root node flow balance:**
\[
\sum_{j: (r,j) \in A} f_{rj} - \sum_{j: (j,r) \in A} f_{jr} = n-1
\]

3. **Non-root node flow balance:**
\[
\sum_{j: (i,j) \in A} f_{ij} - \sum_{j: (j,i) \in A} f_{ji} = -1 \quad \forall i \in N \setminus \{r\}
\]

4. **Flow-to-edge linking:**
\[
f_{ij} \leq M \, y_{kl} \quad \forall (i,j) \in A, \text{ where } \{k,l\} = \{i,j\}
\]

5. **Variable domains:**
\[
y_{ij} \in \{0,1\} \quad \forall \{i,j\} \in E
\]
\[
f_{ij} \geq 0 \quad \forall (i,j) \in A
\]

---

### Data Mapping

- **Nodes $N$:** All "Node" entries in network_nodes.csv (table_id: file_0_view_0)
- **Edges $E$ and costs $c_{ij}$:** All rows in network_edges.csv, with $c_{ij}$ from "ConstructionCost" (table_id: file_1_view_0)
- **Root node $r$:** "Value" where "Parameter" = "RootNode" in network_parameters.csv (table_id: file_2_view_0)
- **Directed arcs $A$:** For each $\{i,j\} \in E$, both $(i,j)$ and $(j,i)$
- **$n$:** Number of rows in network_nodes.csv

All sets, parameters, and constraints are defined directly from the listed CSV files as described above.