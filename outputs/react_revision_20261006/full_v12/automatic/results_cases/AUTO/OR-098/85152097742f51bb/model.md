## Symbolic Mathematical Model

Let $W$ be the set of all workers, as given by the column names (excluding "Owner") in file_0_view_0 (work_days.csv). Let $N = |W|$ (number of workers/homeowners). Let $d_{ij}$ be the number of days worker $j$ worked on homeowner $i$'s home, from file_0_view_0, where $i, j \in W$ and row $i$ corresponds to owner $i$.

Let $x_j$ be the daily wage (yuan) of worker $j \in W$.

The first worker in the file (e.g., "Carpenter") is denoted $w_1$; $x_{w_1} = 60.00$ is fixed.

Each worker contributes exactly 10 work days in total:
$$
\sum_{i \in W} d_{ij} = 10 \qquad \forall j \in W
$$

**Decision variables:**
- $x_j \geq 0$ for all $j \in W$ (daily wage of worker $j$)

**Objective:**
- None (feasibility problem: find $x_j$ satisfying the constraints)

**Constraints:**

1. **Income equals expenditure for each participant:**
   $$
   \sum_{\substack{i \in W \\ i \neq j}} d_{ij} \, x_j = \sum_{\substack{k \in W \\ k \neq j}} d_{jk} \, x_k \qquad \forall j \in W
   $$
   That is, for each $j \in W$:
   - Total income from working on others' homes: $\sum_{i \neq j} d_{ij} x_j$
   - Total expenditure for work performed at their own home: $\sum_{k \neq j} d_{jk} x_k$

   Equivalently, for all $j \in W$:
   $$
   \sum_{i \in W} d_{ij} x_j - d_{jj} x_j = \sum_{k \in W} d_{jk} x_k - d_{jj} x_j
   $$
   $$
   \sum_{i \in W} d_{ij} x_j = \sum_{k \in W} d_{jk} x_k
   $$
   Or, more simply:
   $$
   x_j \sum_{i \in W} d_{ij} = \sum_{k \in W} d_{jk} x_k \qquad \forall j \in W
   $$

   Since $\sum_{i \in W} d_{ij} = 10$ for all $j$, this simplifies to:
   $$
   10 x_j = \sum_{k \in W} d_{jk} x_k \qquad \forall j \in W
   $$

2. **Wage normalization:**
   $$
   x_{w_1} = 60.00
   $$

3. **Nonnegativity:**
   $$
   x_j \geq 0 \qquad \forall j \in W
   $$

## Data Mapping

- $W$: All columns except "Owner" in file_0_view_0 (work_days.csv), table_id: file_0_view_0, columns: as listed.
- $d_{ij}$: Entry in row with Owner $i$, column $j$ in file_0_view_0, table_id: file_0_view_0, columns: as listed.
- $x_j$: Decision variable, daily wage of worker $j$.
- $w_1$: The first worker column in file_0_view_0 (e.g., "Carpenter").

## Complete Model

For all $j \in W$:
$$
10 x_j = \sum_{k \in W} d_{jk} x_k
$$

With
$$
x_{w_1} = 60.00
$$

and
$$
x_j \geq 0 \qquad \forall j \in W
$$

where $d_{jk}$ is from file_0_view_0, row with Owner $j$, column $k$.

**(No objective: feasibility system.)**