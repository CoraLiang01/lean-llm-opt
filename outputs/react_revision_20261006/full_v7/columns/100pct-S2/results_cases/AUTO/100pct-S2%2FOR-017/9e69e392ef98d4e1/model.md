##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from supplier $i$ to customer $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10$\}$ (supplier_id from supply_capacity.csv)
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10$\}$ (customer_id from customer_demand.csv)

##### Parameters

- $d_j$: demand of customer $j$ (from customer_demand.csv)
- $s_i$: supply capacity of supplier $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv, column "transportation_cost_to_Ck" for customer $Ck$)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. Demand satisfaction for each customer:
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$

2. Supply capacity for each supplier:
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i \in I, j \in J
$$

---

##### Data Mapping

- $I$ (suppliers): supplier_id from supply_capacity.csv and transportation_costs.csv
- $J$ (customers): customer_id from customer_demand.csv and transportation_costs.csv
- $d_j$: demand from customer_demand.csv, column "demand", indexed by customer_id
- $s_i$: supply_capacity from supply_capacity.csv, column "supply_capacity", indexed by supplier_id
- $c_{ij}$: transportation_costs.csv, row "supplier_id" $i$, column "transportation_cost_to_Ck" for customer $Ck$ (mapping: C1 $\to$ transportation_cost_to_C1, ..., C10 $\to$ transportation_cost_to_C10)

All indices, parameters, and coefficients are to be taken exactly as listed in the respective CSV files.