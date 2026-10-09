##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from supplier $i$ to customer $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I$: set of suppliers, $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- $J$: set of customers, $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

##### Parameters

- $d_j$: demand of customer $j$ (from "customer_demand.csv")
- $s_i$: supply capacity of supplier $i$ (from "supply_capacity.csv")
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from "transportation_costs.csv")

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   $$
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   $$
2. **Supply capacity:**  
   $$
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

---

##### Data Mapping

- $I$ (suppliers): all "supplier_id" in "supply_capacity.csv" and "transportation_costs.csv" (row axis)
- $J$ (customers): all "customer_id" in "customer_demand.csv" and "transportation_costs.csv" (column axis, mapped via "transportation_cost_to_Ck" columns)
- $d_j$: "demand" column in "customer_demand.csv", indexed by "customer_id"
- $s_i$: "supply_capacity" column in "supply_capacity.csv", indexed by "supplier_id"
- $c_{ij}$: "transportation_cost_to_Ck" columns in "transportation_costs.csv", with $i$ as "supplier_id" (row), $j$ as customer $k$ (column, mapped as per relationships in the Observation)

All indices, parameters, and coefficients are to be taken exactly as listed in the current CSV files and relationships.