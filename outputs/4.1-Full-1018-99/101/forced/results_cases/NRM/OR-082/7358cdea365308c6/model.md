#### Index Sets

- $N$: set of all locations, $N = \{\text{Depot}, A, B, C\}$
- $E$: set of ordered pairs $(i, j)$ where $i, j \in N$, $i \neq j$

#### Parameters

- $d_{ij}$: distance from location $i$ to location $j$  
  (Data: $d_{ij}$ from file_0_view_0, columns "Depot", "A", "B", "C")

#### Decision Variables

- $x_{ij} \in \{0,1\}$:  
  $=1$ if the van travels directly from location $i$ to location $j$, $0$ otherwise, for all $(i, j) \in E$
- $u_i \in \mathbb{Z}$:  
  subtour elimination variable for each $i \in N$, $i \neq \text{Depot}$

#### Objective

Minimize total travel distance:
$$
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} \cdot x_{ij}
$$

#### Constraints

1. **Departure from each node exactly once:**
   $$
   \sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1 \quad \forall i \in N
   $$

2. **Arrival at each node exactly once:**
   $$
   \sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1 \quad \forall j \in N
   $$

3. **Subtour elimination (Miller-Tucker-Zemlin):**
   $$
   u_i - u_j + 3\, x_{ij} \leq 2 \quad \forall i, j \in N,\, i \neq j,\, i \neq \text{Depot},\, j \neq \text{Depot}
   $$
   $$
   1 \leq u_i \leq 3 \quad \forall i \in N,\, i \neq \text{Depot}
   $$

4. **Variable domains:**
   $$
   x_{ij} \in \{0,1\} \quad \forall (i, j) \in E
   $$
   $$
   u_i \in \mathbb{Z} \quad \forall i \in N,\, i \neq \text{Depot}
   $$

---

#### Data Mapping

- Source: file_0_view_0 (DistanceMatrix.csv)
- Node set $N$ and all $d_{ij}$: columns "Depot", "A", "B", "C" and rows with "Unnamed: 0" in {"Depot", "A", "B", "C"}
- $d_{ij}$: value in row where "Unnamed: 0" = $i$, column $j$ (for all $i, j \in N$)

---

This model finds the minimum-distance tour starting and ending at the depot, visiting each of A, B, and C exactly once.