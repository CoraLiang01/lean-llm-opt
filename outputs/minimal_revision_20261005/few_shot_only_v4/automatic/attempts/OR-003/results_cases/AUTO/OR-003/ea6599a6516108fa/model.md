##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from supplier $i$ to customer $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$ (suppliers)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$ (customers)

##### Parameters

- $d_j$: demand of customer $j$ (from customer_demand.csv)
- $s_i$: supply capacity of supplier $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction** (each customer receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   $$
2. **Supply capacity** (each supplier does not exceed its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   $$
3. **Non-negativity**:
   $$
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- **customer_demand.csv**: $d_j$ for $j \in J$
  - $d_{\text{C1}} = 216$
  - $d_{\text{C2}} = 168$
  - $d_{\text{C3}} = 264$
  - $d_{\text{C4}} = 216$
  - $d_{\text{C5}} = 216$
  - $d_{\text{C6}} = 192$
  - $d_{\text{C7}} = 144$
  - $d_{\text{C8}} = 168$
  - $d_{\text{C9}} = 168$
  - $d_{\text{C10}} = 168$

- **supply_capacity.csv**: $s_i$ for $i \in I$
  - $s_{\text{S1}} = 288$
  - $s_{\text{S2}} = 288$
  - $s_{\text{S3}} = 264$
  - $s_{\text{S4}} = 264$
  - $s_{\text{S5}} = 216$
  - $s_{\text{S6}} = 216$
  - $s_{\text{S7}} = 168$
  - $s_{\text{S8}} = 216$
  - $s_{\text{S9}} = 240$
  - $s_{\text{S10}} = 168$

- **transportation_costs.csv**: $c_{ij}$ for $i \in I$, $j \in J$
  - Matrix rows: S1, S2, ..., S10; columns: C1, C2, ..., C10; entries: $c_{ij}$

---

**Summary:**  
Minimize $\sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$  
subject to  
$\sum_{i \in I} x_{ij} \geq d_j$ for all $j \in J$  
$\sum_{j \in J} x_{ij} \leq s_i$ for all $i \in I$  
$x_{ij} \geq 0$ for all $i \in I, j \in J$  
with all parameters and indices as mapped above.