##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i$ to customer group $j$.

- $i \in I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}, \text{S12}\}$
- $j \in J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$

##### Parameters

- Demand $d_j$ for each customer $j$ (from customer_demand.csv)
- Supply capacity $s_i$ for each supplier $i$ (from supply_capacity.csv)
- Transportation cost per unit $c_{ij}$ from supplier $i$ to customer $j$ (from transportation_costs.csv)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:**  
   For each supplier $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- **customer_demand.csv**:  
  - $J = \{\text{C1}, \text{C2}, \ldots, \text{C12}\}$
  - $d_j$ = demand for customer $j$ (column: "demand")

- **supply_capacity.csv**:  
  - $I = \{\text{S1}, \text{S2}, \ldots, \text{S12}\}$
  - $s_i$ = supply capacity for supplier $i$ (column: "supply_capacity")

- **transportation_costs.csv**:  
  - $c_{ij}$ = cost from supplier $i$ (row "Unnamed: 0") to customer $j$ (columns "C1"..."C12")