##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i$ to customer group $j$.

- $i \in I =$ set of distribution centers (from "supply_capacity.csv", column "Unnamed: 0")
- $j \in J =$ set of customer groups (from "customer_demand.csv", column "customer")

##### Objective Function

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$
where $c_{ij}$ is the transportation cost per unit from distribution center $i$ to customer group $j$ (from "transportation_costs.csv", row $i$, column $j$).

##### Constraints

1. **Demand satisfaction:** For each customer group $j \in J$,
$$
\sum_{i \in I} x_{ij} \geq d_j
$$
where $d_j$ is the demand of customer group $j$ (from "customer_demand.csv", column "demand").

2. **Supply capacity:** For each distribution center $i \in I$,
$$
\sum_{j \in J} x_{ij} \leq s_i
$$
where $s_i$ is the supply capacity of distribution center $i$ (from "supply_capacity.csv", column "supply_capacity").

3. **Non-negativity:**
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

##### Data Mapping

- $I$ (distribution centers): all values in "supply_capacity.csv", column "Unnamed: 0", table_id: file_1_view_0
- $J$ (customer groups): all values in "customer_demand.csv", column "customer", table_id: file_0_view_0
- $d_j$: "customer_demand.csv", column "demand", table_id: file_0_view_0, indexed by $j$
- $s_i$: "supply_capacity.csv", column "supply_capacity", table_id: file_1_view_0, indexed by $i$
- $c_{ij}$: "transportation_costs.csv", table_id: file_2_view_0, row index $i$ ("Unnamed: 0"), column index $j$ (customer group ID)

Index sets $I$ and $J$ are defined by the full lists of distribution centers and customer groups in the respective source files. All parameters are mapped directly from the corresponding columns and tables as specified above.