## Minimum-Cost Spanning Tree Model

**Sets**
- $N$: set of nodes (from network_nodes.csv), $N = \{\text{V1}, \text{V2}, \ldots, \text{V9}\}$
- $E$: set of undirected edges (from network_edges.csv), $E = \{(i,j): \text{Node1}=i, \text{Node2}=j\}$
- $A$: set of directed arcs, $A = \{(i,j), (j,i) : (i,j) \in E\}$
- $r$: root node (from network_parameters.csv), $r = \text{V1}$

**Parameters**
- $c_{ij}$: construction cost for undirected edge $(i,j) \in E$ (from network_edges.csv, ConstructionCost)
- $n = |N|$ (number of nodes)
- $M = n-1$

**Variables**
- $y_{ij} \in \{0,1\}$ for $(i,j) \in E$ (edge selection: 1 if edge $(i,j)$ is in the spanning tree)
- $f_{ij} \geq 0$ for $(i,j) \in A$ (flow sent from $i$ to $j$)

**Objective**
\[
\min \sum_{(i,j) \in E} c_{ij} \, y_{ij}
\]

**Constraints**

1. **Edge count (spanning tree):**
   \[
   \sum_{(i,j) \in E} y_{ij} = n-1
   \]

2. **Root node flow balance:**
   \[
   \sum_{j: (r,j) \in A} f_{rj} - \sum_{j: (j,r) \in A} f_{jr} = n-1
   \]

3. **Non-root node flow balance: for all $k \in N \setminus \{r\}$**
   \[
   \sum_{j: (k,j) \in A} f_{kj} - \sum_{j: (j,k) \in A} f_{jk} = -1
   \]

4. **Flow-to-edge linking: for all $(i,j) \in A$**
   \[
   f_{ij} \leq M \cdot y_{e(ij)}
   \]
   where $e(ij)$ is the undirected edge corresponding to arc $(i,j)$ (i.e., $e(ij) = (i,j)$ if $(i,j) \in E$, else $e(ij) = (j,i)$).

5. **Variable domains:**
   \[
   y_{ij} \in \{0,1\} \quad \forall (i,j) \in E
   \]
   \[
   f_{ij} \geq 0 \quad \forall (i,j) \in A
   \]

---

### Data Mapping

- **Node set $N$:** from network_nodes.csv, column "Node", table_id: file_0_view_0
- **Edge set $E$ and costs $c_{ij}$:** from network_edges.csv, columns "Node1", "Node2", "ConstructionCost", table_id: file_1_view_0
- **Root node $r$:** from network_parameters.csv, row with Parameter = "RootNode", column "Value", table_id: file_2_view_0

- **Directed arc set $A$:** for each $(i,j) \in E$, include both $(i,j)$ and $(j,i)$

- **$M = n-1$:** $n$ is the number of nodes in $N$

---

**All sets, parameters, and variables are defined exactly as above, using the full entity lists from the current data.**