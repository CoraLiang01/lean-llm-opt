## Mathematical Model

Let  

* \(N=\{N1,N2,N3,N4,N5,N6,N7,N8\}\) – set of substations (nodes).  
* \(E=\{e_1,\dots ,e_{14}\}\) – set of undirected candidate links.  
  For each \(e\in E\) let \((i_e,j_e)\) be its two endpoints and \(c_e\) its construction cost.  
* \(r=N1\) – root node.  
* \(n=|N|=8\).

### Decision variables
\[
\begin{aligned}
y_e &\in \{0,1\} &&\text{(link \(e\) is built)}\\
f_{ij} &\ge 0 &&\text{(flow on directed arc \(i\!\to\!j\))},\qquad (i,j)\in A
\end{aligned}
\]
where \(A=\{(i_e,j_e),(j_e,i_e)\mid e\in E\}\).

### Objective
\[
\min \;\sum_{e\in E} c_e\,y_e
\]

### Constraints

1. **Tree size**  
   \[
   \sum_{e\in E} y_e = n-1
   \]

2. **Flow conservation**  
   \[
   \sum_{j:(j,i)\in A} f_{ji}-\sum_{j:(i,j)\in A} f_{ij}= 
   \begin{cases}
   1 & i\neq r\\[4pt]
   -(n-1) & i=r
   \end{cases}
   \qquad \forall\, i\in N
   \]

3. **Link‑flow coupling**  
   \[
   f_{ij}\;\le\;(n-1)\,y_e \qquad \forall\,(i,j)\in A,\; e\text{ is the undirected edge between }i\text{ and }j
   \]

4. **Non‑negativity**  
   \[
   f_{ij}\ge 0 \qquad \forall\,(i,j)\in A
   \]

5. **Binary**  
   \[
   y_e\in\{0,1\}\qquad \forall\,e\in E
   \]

---

## Data Mapping

| Parameter | Source | Table ID | Columns | Mapping |
|-----------|--------|----------|---------|---------|
| \(N\) | network_nodes.csv | `file_0_view_0` | `Node` | each row → node identifier |
| \(E\) | network_edges.csv | `file_1_view_0` | `Node1`, `Node2`, `ConstructionCost` | each row → undirected edge \(e\) with endpoints \((i_e,j_e)\) and cost \(c_e\) |
| \(r\) | network_parameters.csv | `file_2_view_0` | `Parameter`, `Value` | row where `Parameter`=`RootNode` → root node \(r\) |
| \(n\) | – | – | – | \(n = |N|\) (computed from \(N\)) |

All indices, parameters, and constraints are defined solely from the data above and the problem statement. No additional data or assumptions are introduced.