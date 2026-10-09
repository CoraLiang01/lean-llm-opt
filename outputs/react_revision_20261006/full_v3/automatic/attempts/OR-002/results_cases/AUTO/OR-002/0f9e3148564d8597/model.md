##### Mathematical Model

Let $I$ be the set of suppliers (stores) and $J$ the set of customer groups, as defined by the data.

**Decision Variables:**

For each $i\in I$, $j\in J$:
$$
x_{ij} \geq 0
$$
where $x_{ij}$ is the quantity shipped from supplier $i$ to customer $j$ (continuous).

**Parameters:**

- $d_j$: demand of customer $j$ (from `customer_demand.csv`)
- $s_i$: supply capacity of supplier $i$ (from `supply_capacity.csv`)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from `transportation_costs.csv`)

**Objective:**

Minimize total transportation cost:
$$
\min \sum_{i\in I} \sum_{j\in J} c_{ij} x_{ij}
$$

**Constraints:**

1. **Demand satisfaction:** For each customer $j\in J$,
$$
\sum_{i\in I} x_{ij} \geq d_j
$$

2. **Supply capacity:** For each supplier $i\in I$,
$$
\sum_{j\in J} x_{ij} \leq s_i
$$

3. **Non-negativity:** For all $i\in I$, $j\in J$,
$$
x_{ij} \geq 0
$$

##### Data Mapping

- $I$ (suppliers): All values in column `Unnamed: 0` of `file_1_view_0` (supply_capacity.csv)
- $J$ (customers): All values in column `customer` of `file_0_view_0` (customer_demand.csv)
- $d_j$: For each $j\in J$, value in column `demand` of `file_0_view_0` where `customer` = $j$
- $s_i$: For each $i\in I$, value in column `supply_capacity` of `file_1_view_0` where `Unnamed: 0` = $i$
- $c_{ij}$: For each $i\in I$, $j\in J$, value in column $j$ of `file_2_view_0` (transportation_costs.csv) where `Unnamed: 0` = $i$ and column = $j$
- $x_{ij}$: Decision variable for each $i\in I$, $j\in J$ (continuous, $\geq 0$)

**Index sets $I$ and $J$ must include all entities present in the respective source columns.**