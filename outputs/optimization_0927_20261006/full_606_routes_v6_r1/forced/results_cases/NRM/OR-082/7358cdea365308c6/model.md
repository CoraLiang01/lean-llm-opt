#### Index Sets

- $N$: set of all locations, $N = \{\text{Depot}, A, B, C\}$
- $V$: set of customer locations, $V = \{A, B, C\}$

#### Parameters

- $d_{ij}$: distance from location $i$ to location $j$, for all $i, j \in N$

#### Decision Variables

- $x_{ij} \in \{0,1\}$: equals 1 if the route goes directly from location $i$ to location $j$, 0 otherwise, for all $i, j \in N$, $i \neq j$
- $u_i \in \mathbb{Z}$: auxiliary variables for subtour elimination, for all $i \in V$

#### Objective

Minimize the total travel distance:
$$
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} x_{ij}
$$

#### Constraints

1. **Departure from Depot:**
   $$
   \sum_{\substack{j \in N \\ j \neq \text{Depot}}} x_{\text{Depot},j} = 1
   $$
2. **Return to Depot:**
   $$
   \sum_{\substack{i \in N \\ i \neq \text{Depot}}} x_{i,\text{Depot}} = 1
   $$
3. **Visit Each Customer Exactly Once:**
   $$
   \sum_{\substack{i \in N \\ i \neq k}} x_{ik} = 1, \quad \forall k \in V
   $$
   $$
   \sum_{\substack{j \in N \\ j \neq k}} x_{kj} = 1, \quad \forall k \in V
   $$
4. **Subtour Elimination (MTZ Formulation):**
   $$
   u_i - u_j + 3 x_{ij} \leq 2, \quad \forall i, j \in V,\, i \neq j
   $$
   $$
   1 \leq u_i \leq 3, \quad \forall i \in V
   $$

5. **Variable Domains:**
   $$
   x_{ij} \in \{0,1\}, \quad \forall i, j \in N,\, i \neq j
   $$

#### Data Mapping

- Table: DistanceMatrix.csv (table_id: file_0_view_0)
- Columns: "Unnamed: 0" (row label: location $i$), "Depot", "A", "B", "C" (columns: location $j$)
- For all $i, j \in N$, $d_{ij}$ is the value in row where "Unnamed: 0" = $i$ and column $j$.

---

This abstract model symbolically defines the Traveling Salesman Problem for the depot and three customers, using the provided distance matrix. The optimal sequence of visits is determined by solving this model with the supplied data.