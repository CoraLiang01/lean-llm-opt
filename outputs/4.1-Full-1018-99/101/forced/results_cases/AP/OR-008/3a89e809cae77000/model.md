##### Sets

- Suppliers: $\{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$
- Customers: $\{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$

##### Parameters

- Customer demands:
  - $d_{\text{Customer1}} = 70$
  - $d_{\text{Customer2}} = 80$
  - $d_{\text{Customer3}} = 60$
  - $d_{\text{Customer4}} = 90$
  - $d_{\text{Customer5}} = 85$
  - $d_{\text{Customer6}} = 95$

- Supplier capacities:
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

- $x_{ij} \geq 0$: Amount of produce shipped from supplier $i$ to customer $j$.

##### Objective Function

$\min \sum_{i \in \text{Suppliers}} \sum_{j \in \text{Customers}} c_{ij} x_{ij}$

##### Constraints

1. **Demand Satisfaction (for each customer):**

$\sum_{i \in \text{Suppliers}} x_{ij} = d_j \quad \forall j \in \text{Customers}$

2. **Supply Capacity (for each supplier):**

$\sum_{j \in \text{Customers}} x_{ij} \leq s_i \quad \forall i \in \text{Suppliers}$

3. **Non-negativity:**

$x_{ij} \geq 0 \quad \forall i \in \text{Suppliers},\ \forall j \in \text{Customers}$

##### Retrieved Information

{
  "customer_demand": {
    "Customer1": 70,
    "Customer2": 80,
    "Customer3": 60,
    "Customer4": 90,
    "Customer5": 85,
    "Customer6": 95
  },
  "supply_capacity": {
    "Supplier1": 200,
    "Supplier2": 250,
    "Supplier3": 230,
    "Supplier4": 220,
    "Supplier5": 210
  },
  "transportation_cost": {
    "Supplier1": {
      "Customer1": 2,
      "Customer2": 3,
      "Customer3": 1,
      "Customer4": 2,
      "Customer5": 3,
      "Customer6": 2
    },
    "Supplier2": {
      "Customer1": 1,
      "Customer2": 2,
      "Customer3": 3,
      "Customer4": 2,
      "Customer5": 3,
      "Customer6": 2
    },
    "Supplier3": {
      "Customer1": 3,
      "Customer2": 1,
      "Customer3": 2,
      "Customer4": 3,
      "Customer5": 2,
      "Customer6": 3
    },
    "Supplier4": {
      "Customer1": 2,
      "Customer2": 3,
      "Customer3": 2,
      "Customer4": 1,
      "Customer5": 3,
      "Customer6": 4
    },
    "Supplier5": {
      "Customer1": 3,
      "Customer2": 2,
      "Customer3": 3,
      "Customer4": 3,
      "Customer5": 2,
      "Customer6": 3
    }
  }
}