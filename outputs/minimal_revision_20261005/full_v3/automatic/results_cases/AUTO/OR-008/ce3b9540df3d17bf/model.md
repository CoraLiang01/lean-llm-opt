##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of fresh produce shipped from warehouse (supplier) $i$ to store (customer) $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I$: Set of warehouses (suppliers), from file_1_view_0:  
  $I = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$
- $J$: Set of stores (customers), from file_0_view_0:  
  $J = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$

##### Parameters

- $d_j$: Demand of store $j$, from file_0_view_0, column "demand"
- $s_i$: Supply capacity of warehouse $i$, from file_1_view_0, column "supply_capacity"
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to store $j$, from file_2_view_0, row "Unnamed: 0" (supplier), columns "Customer1"–"Customer6"

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:**  
   For each warehouse $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- **Demands $d_j$** (from file_0_view_0, column "demand"):
  - $d_{\text{Customer1}} = 70$
  - $d_{\text{Customer2}} = 80$
  - $d_{\text{Customer3}} = 60$
  - $d_{\text{Customer4}} = 90$
  - $d_{\text{Customer5}} = 85$
  - $d_{\text{Customer6}} = 95$

- **Supply capacities $s_i$** (from file_1_view_0, column "supply_capacity"):
  - $s_{\text{Supplier1}} = 200$
  - $s_{\text{Supplier2}} = 250$
  - $s_{\text{Supplier3}} = 230$
  - $s_{\text{Supplier4}} = 220$
  - $s_{\text{Supplier5}} = 210$

- **Transportation costs $c_{ij}$** (from file_2_view_0, row "Unnamed: 0" for $i$, columns "Customer1"–"Customer6" for $j$):

  | $c_{ij}$         | Customer1 | Customer2 | Customer3 | Customer4 | Customer5 | Customer6 |
  |------------------|-----------|-----------|-----------|-----------|-----------|-----------|
  | Supplier1        |     2     |     3     |     1     |     2     |     3     |     2     |
  | Supplier2        |     1     |     2     |     3     |     2     |     3     |     2     |
  | Supplier3        |     3     |     1     |     2     |     3     |     2     |     3     |
  | Supplier4        |     2     |     3     |     2     |     1     |     3     |     4     |
  | Supplier5        |     3     |     2     |     3     |     3     |     2     |     3     |

  (Table indices: file_2_view_0, row "Unnamed: 0" for suppliers, columns "Customer1"–"Customer6" for customers.)

---

#### Complete Model

Minimize
$$
\sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

subject to
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

with all parameters and indices as mapped above.