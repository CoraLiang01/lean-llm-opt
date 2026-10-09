##### Parameters

- Warehouses (Suppliers): $I = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$
- Stores (Customers): $J = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$

- Store demands ($d_j$):
  - $d_{\text{Customer1}} = 70$
  - $d_{\text{Customer2}} = 80$
  - $d_{\text{Customer3}} = 60$
  - $d_{\text{Customer4}} = 90$
  - $d_{\text{Customer5}} = 85$
  - $d_{\text{Customer6}} = 95$

- Warehouse supply capacities ($s_i$):
  - $s_{\text{Supplier1}} = 200$
  - $s_{\text{Supplier2}} = 250$
  - $s_{\text{Supplier3}} = 230$
  - $s_{\text{Supplier4}} = 220$
  - $s_{\text{Supplier5}} = 210$

- Transportation costs per unit ($c_{ij}$):

|            | Customer1 | Customer2 | Customer3 | Customer4 | Customer5 | Customer6 |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Supplier1  |     2     |     3     |     1     |     2     |     3     |     2     |
| Supplier2  |     1     |     2     |     3     |     2     |     3     |     2     |
| Supplier3  |     3     |     1     |     2     |     3     |     2     |     3     |
| Supplier4  |     2     |     3     |     2     |     1     |     3     |     4     |
| Supplier5  |     3     |     2     |     3     |     3     |     2     |     3     |

##### Decision Variables

- $x_{ij} \geq 0$: Amount of fresh produce shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction at each store:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supply capacity at each warehouse:**
   \[
   \sum_{j \in J} x_{ij} \leq s_i, \quad \forall i \in I
   \]

3. **Nonnegativity:**
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]

##### Full Parameter Listing

- $I = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$
- $J = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$
- $d_j$ as above
- $s_i$ as above
- $c_{ij}$ as above

This model determines the optimal shipment plan from warehouses to stores, meeting all demands and not exceeding any warehouse's capacity, while minimizing total transportation cost.