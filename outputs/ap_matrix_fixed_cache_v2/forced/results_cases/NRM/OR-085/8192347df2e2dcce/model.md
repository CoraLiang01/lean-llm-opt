#### Index Sets

- $N$: set of locations (customers), indexed by $i, j$. Here, $N = \{1, 2, \ldots, 15\}$.

#### Parameters

- $d_{ij}$: distance from location $i$ to location $j$.
    - Data Mapping: $d_{ij}$ is given by the entry in table_id file_0_view_0, row with Unnamed: 0 = $i$, column $j$.

#### Variables

- $x_{ij} \in \{0,1\}$: equals 1 if the tour goes directly from location $i$ to location $j$, 0 otherwise, for all $i, j \in N$, $i \neq j$.
- $u_i \in \mathbb{Z}$, $2 \leq u_i \leq |N|$, for all $i \in N$, $i \neq 1$: auxiliary variables for subtour elimination (Miller-Tucker-Zemlin).

#### Objective

$$
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} \cdot x_{ij}
$$

#### Constraints

1. **Departure Constraint:** Each location is departed from exactly once:
   $$
   \sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1, \quad \forall i \in N
   $$

2. **Arrival Constraint:** Each location is arrived at exactly once:
   $$
   \sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1, \quad \forall j \in N
   $$

3. **Subtour Elimination (MTZ):** For all $i, j \in N$, $i \neq j$, $i \neq 1$, $j \neq 1$:
   $$
   u_i - u_j + |N| \cdot x_{ij} \leq |N| - 1
   $$
   $$
   2 \leq u_i \leq |N|, \quad \forall i \in N, i \neq 1
   $$

4. **Start and End at Location 1:** The tour starts and ends at location 1 (enforced by the degree constraints above).

5. **Variable Domains:**
   $$
   x_{ij} \in \{0,1\}, \quad \forall i, j \in N, i \neq j
   $$
   $$
   u_i \in \mathbb{Z}, \quad 2 \leq u_i \leq |N|, \quad \forall i \in N, i \neq 1
   $$

---

#### Data Mapping

- $d_{ij}$: file_0_view_0, column "Unnamed: 0" (row index $i$), column "$j$" (column index $j$), for all $i, j \in N$.
- $N$: All unique values in column "Unnamed: 0" of file_0_view_0, and all column names "1" through "15" in file_0_view_0.

No additional filtering or selection was applied; the full distance matrix was used as returned by CSVQA.