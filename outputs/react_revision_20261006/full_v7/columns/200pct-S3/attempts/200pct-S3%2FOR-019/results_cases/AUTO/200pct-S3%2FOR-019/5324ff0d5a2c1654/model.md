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

1. **Demand satisfaction:**  
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity:**  
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

##### Data Mapping

- $I$ (suppliers): all unique values in column "supplier_id" of supply_capacity.csv and transportation_costs.csv
- $J$ (customer groups): all unique values in column "customer_id" of customer_demand.csv and as suffixes in transportation_costs.csv columns
- $d_j$: parameter from customer_demand.csv, column "demand", indexed by "customer_id"
- $s_i$: parameter from supply_capacity.csv, column "supply_capacity", indexed by "supplier_id"
- $c_{ij}$: parameter from transportation_costs.csv, value in column "transportation_cost_to_{j}" for row with "supplier_id" = $i$, for each $i \in I$, $j \in J$

All indices, parameters, and coefficients are mapped exactly as in the retrieved data.