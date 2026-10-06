##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of fresh produce shipped from warehouse (supplier) $i$ to store (customer) $j$, for all $i \in I$, $j \in J$.

Where:
- $I = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$
- $J = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$

##### Parameters

- $d_j$: demand of customer $j$
- $s_i$: supply capacity of supplier $i$
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$

##### Objective Function

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

- **Customers $J$ and Demands $d_j$** (from customer_demand.csv, in source order):

  | Customer    | $d_j$ |
  |-------------|-------|
  | Customer1   | 70    |
  | Customer2   | 80    |
  | Customer3   | 60    |
  | Customer4   | 90    |
  | Customer5   | 85    |
  | Customer6   | 95    |

- **Suppliers $I$ and Supply Capacities $s_i$** (from supply_capacity.csv, in source order):

  | Supplier    | $s_i$ |
  |-------------|-------|
  | Supplier1   | 200   |
  | Supplier2   | 250   |
  | Supplier3   | 230   |
  | Supplier4   | 220   |
  | Supplier5   | 210   |

- **Transportation Costs $c_{ij}$** (from transportation_costs.csv, rows: suppliers in source order, columns: customers in source order):

  |            | Customer1 | Customer2 | Customer3 | Customer4 | Customer5 | Customer6 |
  |------------|-----------|-----------|-----------|-----------|-----------|-----------|
  | Supplier1  | 2         | 3         | 1         | 2         | 3         | 2         |
  | Supplier2  | 1         | 2         | 3         | 2         | 3         | 2         |
  | Supplier3  | 3         | 1         | 2         | 3         | 2         | 3         |
  | Supplier4  | 2         | 3         | 2         | 1         | 3         | 4         |
  | Supplier5  | 3         | 2         | 3         | 3         | 2         | 3         |