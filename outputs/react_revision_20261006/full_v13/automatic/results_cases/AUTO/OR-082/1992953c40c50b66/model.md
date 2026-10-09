## Mathematical Model

**Sets**

- $N = \{\text{Depot}, A, B, C\}$: set of all locations (including depot and customers)
- $V = \{A, B, C\}$: set of customer locations

**Parameters**

- $d_{ij}$: distance from location $i$ to location $j$ (in km), for all $i, j \in N$, from DistanceMatrix.csv (table_id: file_0_view_0, columns: Unnamed: 0 as $i$, [Depot, A, B, C] as $j$)

**Decision Variables**

- $x_{ij} \in \{0,1\}$: 1 if the van travels directly from $i$ to $j$, 0 otherwise, for all $i, j \in N$, $i \neq j$

**Objective**

Minimize the total travel distance:
$$
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} x_{ij}
$$

**Constraints**

1. **Each location (except depot) is entered exactly once:**
   $$
   \sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1 \quad \forall j \in V
   $$
2. **Each location (except depot) is left exactly once:**
   $$
   \sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1 \quad \forall i \in V
   $$
3. **Depot is left once and entered once:**
   $$
   \sum_{j \in V} x_{\text{Depot},j} = 1
   $$
   $$
   \sum_{i \in V} x_{i,\text{Depot}} = 1
   $$
4. **Subtour elimination (MTZ formulation):**
   Introduce continuous variables $u_i$ for $i \in V$, with $1 \leq u_i \leq 3$.
   $$
   u_i - u_j + 3 x_{ij} \leq 2 \quad \forall i, j \in V,\, i \neq j
   $$
5. **Variable domains:**
   $$
   x_{ij} \in \{0,1\} \quad \forall i, j \in N,\, i \neq j
   $$
   $$
   u_i \in [1,3] \quad \forall i \in V
   $$

**Data Mapping**

- $d_{ij}$: DistanceMatrix.csv, table_id: file_0_view_0, with $i$ from column "Unnamed: 0" and $j$ from columns "Depot", "A", "B", "C".

**Summary**

This is a standard symmetric Traveling Salesman Problem (TSP) for 4 nodes (Depot, A, B, C), with the objective to minimize total travel distance, visiting each customer exactly once and returning to the depot. The MTZ constraints prevent subtours. All parameters and sets are mapped directly from the provided DistanceMatrix.csv.