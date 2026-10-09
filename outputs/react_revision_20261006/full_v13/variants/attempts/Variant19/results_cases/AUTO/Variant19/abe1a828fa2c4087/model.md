## Minimum-Cost Spanning Tree Model

**Sets**
- $N$: set of nodes (from network_nodes.csv), indexed by $i,j$.
- $E$: set of undirected edges (from network_edges.csv), each edge $e = \{i,j\}$ with $i < j$.
- $A$: set of directed arcs, i.e., for each $\{i,j\} \in E$, both $(i,j)$ and $(j,i)$.
- $r$: root node (from network_parameters.csv, $r = \text{V1}$).

**Parameters**
- $c_{ij}$: construction cost for undirected edge $\{i,j\} \in E$ (from network_edges.csv).
- $n$: $|N|$ (number of nodes, here $n=9$).
- $M = n-1$.

**Variables**
- $y_{ij} \in \{0,1\}$: $1$ if undirected edge $\{i,j\} \in E$ is selected, $0$ otherwise.
- $f_{ij} \geq 0$: flow on directed arc $(i,j) \in A$ (continuous, nonnegative).

**Objective**
\[
\min \sum_{\{i,j\} \in E} c_{ij} \, y_{ij}
\]

**Constraints**

1. **Edge Count**
   \[
   \sum_{\{i,j\} \in E} y_{ij} = n-1
   \]

2. **Root Node Flow Balance** (for $r$)
   \[
   \sum_{j: (r,j) \in A} f_{rj} - \sum_{j: (j,r) \in A} f_{jr} = n-1
   \]

3. **Non-root Node Flow Balance** (for all $i \in N \setminus \{r\}$)
   \[
   \sum_{j: (i,j) \in A} f_{ij} - \sum_{j: (j,i) \in A} f_{ji} = -1
   \]

4. **Flow-to-Edge Linking** (for all $(i,j) \in A$ corresponding to $\{i,j\} \in E$)
   \[
   f_{ij} \leq M \, y_{ij}
   \]

5. **Variable Domains**
   \[
   y_{ij} \in \{0,1\} \quad \forall \{i,j\} \in E
   \]
   \[
   f_{ij} \geq 0 \quad \forall (i,j) \in A
   \]

---

### Data Mapping

- **Nodes ($N$):** All "Node" entries in network_nodes.csv (table_id: file_0_view_0).
- **Edges ($E$):** All pairs $\{i,j\}$ from "Node1", "Node2" in network_edges.csv (table_id: file_1_view_0).
- **Edge Costs ($c_{ij}$):** "ConstructionCost" for each $\{i,j\}$ in network_edges.csv (table_id: file_1_view_0).
- **Root Node ($r$):** "Value" where "Parameter" = "RootNode" in network_parameters.csv (table_id: file_2_view_0).
- **Directed Arcs ($A$):** For each $\{i,j\} \in E$, both $(i,j)$ and $(j,i)$.
- **$n$:** Number of rows in network_nodes.csv.
- **$M$:** $n-1$.

---

**All sets, parameters, and variables are defined exactly as above, using the full entity lists from the current data.**