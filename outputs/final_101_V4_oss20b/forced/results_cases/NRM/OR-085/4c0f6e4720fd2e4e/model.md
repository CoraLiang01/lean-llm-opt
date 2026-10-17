**Symbolic Model**

Let  

* \(N=\{1,\dots ,15\}\) – set of locations (customer 1 is the depot).  
* \(d_{ij}\) – distance from location \(i\) to location \(j\) (data from *20.csv*).  
* \(x_{ij}\in\{0,1\}\) – binary variable, 1 if the tour goes directly from \(i\) to \(j\).  
* \(u_i\in\{2,\dots ,15\}\) – auxiliary MTZ variables for subtour elimination (only for \(i\neq1\)).  

**Objective**

\[
\min \;\; \sum_{i\in N}\sum_{j\in N} d_{ij}\,x_{ij}
\]

**Constraints**

1. **Out‑degree (depot)**  
   \[
   \sum_{j\in N\setminus\{1\}} x_{1j}=1
   \]

2. **In‑degree (depot)**  
   \[
   \sum_{i\in N\setminus\{1\}} x_{i1}=1
   \]

3. **Out‑degree (other nodes)**  
   \[
   \sum_{j\in N\setminus\{i\}} x_{ij}=1 \quad \forall i\in N\setminus\{1\}
   \]

4. **In‑degree (other nodes)**  
   \[
   \sum_{i\in N\setminus\{j\}} x_{ij}=1 \quad \forall j\in N\setminus\{1\}
   \]

5. **MTZ subtour elimination**  
   \[
   u_i - u_j + 14\,x_{ij} \le 13 \quad \forall i,j\in N\setminus\{1\},\; i\neq j
   \]

6. **Variable domains**  
   \[
   x_{ij}\in\{0,1\}\;\forall i,j\in N,\qquad
   u_i\in\{2,\dots ,15\}\;\forall i\in N\setminus\{1\}
   \]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(N\) | implicit | \(N=\{1,\dots ,15\}\) |
| \(d_{ij}\) | *20.csv* (table_id `file_0_view_0`) | \(d_{ij}\) equals the value in column **j** of row **i** (both 1‑based indices). |
| \(x_{ij}\) | decision variable | binary variable for each ordered pair \((i,j)\in N\times N\). |
| \(u_i\) | auxiliary variable | integer variable for each \(i\in N\setminus\{1\}\). |

*All other data (objective sense, variable domains, constraints) are derived from the problem statement and not directly from the CSV.*