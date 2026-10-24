**Symbolic Model**

Let  

* \(N=\{Depot, A, B, C\}\) – set of nodes (depot + customers).  
* \(d_{ij}\) – distance from node \(i\) to node \(j\) (km).  
* \(x_{ij}\in\{0,1\}\) – binary variable, 1 if the vehicle travels directly from \(i\) to \(j\).  
* \(u_i\in\{1,\dots,|N|-1\}\) – MTZ auxiliary variables for nodes \(i\in N\setminus\{Depot\}\).

---

### Objective
\[
\min \;\; \sum_{i\in N}\sum_{j\in N} d_{ij}\,x_{ij}
\]

### Constraints

1. **Out‑degree (each node leaves once)**  
   \[
   \sum_{j\in N} x_{ij}=1 \quad \forall i\in N
   \]

2. **In‑degree (each node entered once)**  
   \[
   \sum_{i\in N} x_{ij}=1 \quad \forall j\in N
   \]

3. **MTZ subtour elimination**  
   \[
   u_i - u_j + (|N|-1)\,x_{ij} \le |N|-2 \quad \forall i,j\in N\setminus\{Depot\},\; i\neq j
   \]

4. **Variable domains**  
   \[
   x_{ij}\in\{0,1\}\quad \forall i,j\in N
   \]
   \[
   u_i\in\{1,\dots,|N|-1\}\quad \forall i\in N\setminus\{Depot\}
   \]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(d_{ij}\) | `file_0_view_0` (DistanceMatrix.csv) | For each pair \((i,j)\in N\times N\), set \(d_{ij}\) to the value in the cell where the row header equals \(i\) and the column header equals \(j\). The table has rows and columns labeled `Depot`, `A`, `B`, `C`. |
| \(N\) | Derived from the row/column headers of `file_0_view_0` | \(N=\{\text{Depot},\text{A},\text{B},\text{C}\}\) |

No other data are required. The model above fully specifies the traveling‑salesman problem for the courier van and can be solved to obtain the optimal visiting sequence that minimizes total travel distance.