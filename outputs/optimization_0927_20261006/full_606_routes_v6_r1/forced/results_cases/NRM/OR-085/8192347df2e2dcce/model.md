#### Index Sets

- $N$: set of locations (customers), indexed by $i, j$. Here, $N = \{1, 2, \ldots, 15\}$.

#### Parameters

- $d_{ij}$: distance from location $i$ to location $j$ for all $i, j \in N$, $i \neq j$.

#### Decision Variables

- $x_{ij} \in \{0,1\}$: equals 1 if the tour goes directly from location $i$ to location $j$, 0 otherwise, for all $i, j \in N$, $i \neq j$.
- $u_i \in \mathbb{Z}$, $2 \leq u_i \leq |N|$, for all $i \in N$, $i \neq 1$: auxiliary variables for subtour elimination (Miller-Tucker-Zemlin formulation).

#### Objective

Minimize the total distance traveled:
$$
\min \sum_{i \in N} \sum_{j \in N,\, j \neq i} d_{ij} x_{ij}
$$

#### Constraints

1. **Departure Constraint:** Leave each location exactly once.
   $$
   \sum_{j \in N,\, j \neq i} x_{ij} = 1 \quad \forall i \in N
   $$

2. **Arrival Constraint:** Enter each location exactly once.
   $$
   \sum_{i \in N,\, i \neq j} x_{ij} = 1 \quad \forall j \in N
   $$

3. **Subtour Elimination (MTZ):** For all $i, j \in N$, $i \neq j$, $i \neq 1$, $j \neq 1$:
   $$
   u_i - u_j + |N|\, x_{ij} \leq |N| - 1
   $$

4. **Start and End at Location 1:** The tour starts and ends at location 1 (enforced by the above constraints).

5. **Variable Domains:**
   $$
   x_{ij} \in \{0,1\} \quad \forall i, j \in N,\, i \neq j
   $$
   $$
   u_i \in \mathbb{Z},\ 2 \leq u_i \leq |N| \quad \forall i \in N,\, i \neq 1
   $$

---

#### Data Mapping

- Table: `file_0_view_0` (from `20.csv`)
- Row and column identifiers: `Unnamed: 0` (location indices $1$ to $15$)
- Distance parameter: $d_{ij}$ is given by the value in row with `Unnamed: 0 = i` and column with header $j$ (for $i \neq j$), from columns `"1"` to `"15"`.
- The full $15 \times 15$ symmetric distance matrix is used as returned by the query.