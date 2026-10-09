##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from distribution center (supplier) $i$ to customer group $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

##### Parameters

- $d_j$: demand of customer group $j$ (from customer_demand.csv)
- $s_i$: supply capacity of supplier $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer group $j$ (from transportation_costs.csv)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** For each customer group $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:** For each supplier $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (suppliers): supplier_id from supply_capacity.csv and transportation_costs.csv rows
- $J$ (customer groups): customer_id from customer_demand.csv and transportation_costs.csv columns (suffix after "transportation_cost_to_")
- $d_j$: demand from customer_demand.csv, column "demand", indexed by "customer_id"
- $s_i$: supply_capacity from supply_capacity.csv, column "supply_capacity", indexed by "supplier_id"
- $c_{ij}$: transportation_costs.csv, value at row with supplier_id $i$ and column "transportation_cost_to_$j$" (where $j$ matches customer_id)

All indices, parameters, and coefficients are to be taken exactly as in the source files and relationships above.