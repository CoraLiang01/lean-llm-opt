## Minimum-Cost Spanning Tree Model

**Sets**
- $N$: set of nodes (from network_nodes.csv), indexed by $i,j$
- $E$: set of undirected edges (from network_edges.csv), each edge $e = \{i,j\}$ with $i < j$
- $A$: set of directed arcs, i.e., for each $\{i,j\} \in E$, both $(i,j)$ and $(j,i)$
- $r$: root node (from network_parameters.csv, $r = $ V1)

**Parameters**
- $c_{ij}$: construction cost for undirected edge $\{i,j\}$ (from network_edges.csv)
- $n$: $|N|$ (number of nodes)
- $M$: $n-1$

**Variables**
- $y_{ij} \in \{0,1\}$ for each $\{i,j\} \in E$: 1 if edge $\{i,j\}$ is selected, 0 otherwise
- $f_{ij} \geq 0$ for each $(i,j) \in A$: flow sent from $i$ to $j$ (continuous, nonnegative)

**Objective**
Minimize total construction cost:
\[
\min \sum_{\{i,j\} \in E} c_{ij} y_{ij}
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
\forall i \in N \setminus \{r\}:\quad \sum_{j: (i,j) \in A} f_{ij} - \sum_{j: (j,i) \in A} f_{ji} = -1
\]

4. **Flow-to-edge linking:**
\[
\forall \{i,j\} \in E:
\begin{cases}
f_{ij} \leq (n-1) y_{ij} \\
f_{ji} \leq (n-1) y_{ij}
\end{cases}
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

- $N$ (nodes): all "Node" values in network_nodes.csv (table_id: file_0_view_0)
- $E$ (edges): all pairs $\{i,j\}$ from "Node1", "Node2" in network_edges.csv (table_id: file_1_view_0)
- $c_{ij}$: "ConstructionCost" for each $\{i,j\}$ in network_edges.csv (table_id: file_1_view_0)
- $A$ (arcs): for each $\{i,j\} \in E$, both $(i,j)$ and $(j,i)$
- $r$: "Value" where "Parameter" = "RootNode" in network_parameters.csv (table_id: file_2_view_0)
- $n$: number of rows in network_nodes.csv

All sets, parameters, and constraints are defined using the full current data from the specified CSV files.