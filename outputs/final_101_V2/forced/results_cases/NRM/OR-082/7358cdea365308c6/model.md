#### Abstract Mathematical Model

**Index Sets:**
- $N$: Set of all locations, $N = \{\text{Depot}, A, B, C\}$

**Parameters:**
- $d_{ij}$: Distance from location $i$ to location $j$, for all $i, j \in N$  
  (from DistanceMatrix.csv, columns and rows: Depot, A, B, C)

**Decision Variables:**
- $x_{ij} \in \{0,1\}$:  
  $x_{ij} = 1$ if the van travels directly from location $i$ to location $j$; $0$ otherwise, for all $i, j \in N$, $i \neq j$

- $u_i \in \mathbb{Z}$:  
  Subtour elimination variable for each $i \in N$, $i \neq \text{Depot}$

**Objective:**
\[
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} \cdot x_{ij}
\]

**Constraints:**

1. **Departure from Each Location:**
   \[
   \sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1 \quad \forall i \in N
   \]

2. **Arrival to Each Location:**
   \[
   \sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1 \quad \forall j \in N
   \]

3. **Subtour Elimination (MTZ):**
   \[
   u_i - u_j + 3 x_{ij} \leq 2 \quad \forall i, j \in N,\, i \neq j,\, i \neq \text{Depot},\, j \neq \text{Depot}
   \]
   \[
   1 \leq u_i \leq 3 \quad \forall i \in N,\, i \neq \text{Depot}
   \]

4. **Variable Domains:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i, j \in N,\, i \neq j
   \]
   \[
   u_i \in \mathbb{Z} \quad \forall i \in N,\, i \neq \text{Depot}
   \]

---

#### Data Mapping

- **Table:** DistanceMatrix.csv
- **Table ID:** file_0_view_0
- **Columns:** "Depot", "A", "B", "C"
- **Row Identifiers:** "Depot", "A", "B", "C"
- **Parameter Mapping:** $d_{ij}$ is the value in row $i$, column $j$ of DistanceMatrix.csv, for all $i, j \in N$.

---

**Note:**  
This is a standard TSP (Travelling Salesman Problem) formulation for four nodes (Depot, A, B, C) with pairwise distances as parameters. The optimal sequence of visits is determined by solving this model with the provided distance matrix.