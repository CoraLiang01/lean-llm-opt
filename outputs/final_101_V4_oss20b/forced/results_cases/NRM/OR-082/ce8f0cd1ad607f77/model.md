**Abstract Model**

**Sets**  
- \(N=\{0,1,2,3\}\) (0 = Depot, 1 = A, 2 = B, 3 = C)  
- \(A=\{(i,j)\mid i,j\in N,\;i\neq j\}\) (all directed arcs)

**Parameters**  
- \(d_{ij}\) = distance between node \(i\) and node \(j\) (read from *DistanceMatrix.csv*).

**Decision Variables**  
- \(x_{ij}\in\{0,1\}\) = 1 if the van travels directly from node \(i\) to node \(j\), 0 otherwise.  
- \(u_i\in\mathbb{Z}\) = auxiliary variable for subtour elimination (Miller‑Tucker‑Zemlin), with \(u_0=0\).

**Objective**  
$$
\min \sum_{(i,j)\in A} d_{ij}\,x_{ij}
$$

**Constraints**

1. **Out‑degree of each node**  
$$
\sum_{j\in N,\;j\neq i} x_{ij}=1 \quad \forall i\in N
$$

2. **In‑degree of each node**  
$$
\sum_{i\in N,\;i\neq j} x_{ij}=1 \quad \forall j\in N
$$

3. **Subtour elimination (Miller‑Tucker‑Zemlin)**  
$$
u_i-u_j+4\,x_{ij}\le 3 \quad \forall i,j\in N,\;i\neq j,\;i\neq 0,\;j\neq 0
$$

4. **Auxiliary variable bounds**  
$$
1\le u_i\le 3 \quad \forall i\in N,\;i\neq 0
$$

5. **Variable domains**  
$$
x_{ij}\in\{0,1\}\quad \forall (i,j)\in A
$$

---

**Data Mapping**

| Source | Table ID | Columns Used | Description |
|--------|----------|--------------|-------------|
| Distance data | `file_0_view_0` | `Depot`, `A`, `B`, `C` | Provides the pairwise distances \(d_{ij}\) between the depot and the three customer locations. The matrix is read from *DistanceMatrix.csv* and used to populate the parameter \(d_{ij}\). |

The model above fully specifies the travelling‑salesman problem for the courier van. Solving it (e.g., with an MILP solver) yields the optimal visitation sequence that minimises total travel distance.