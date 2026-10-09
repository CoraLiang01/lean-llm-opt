##### Sets

- Let $S = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$ be the set of suppliers (warehouses).
- Let $C = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$ be the set of customers (stores).

##### Parameters

- Demand for each customer:
  - $d_{\text{Customer1}} = 70$
  - $d_{\text{Customer2}} = 80$
  - $d_{\text{Customer3}} = 60$
  - $d_{\text{Customer4}} = 90$
  - $d_{\text{Customer5}} = 85$
  - $d_{\text{Customer6}} = 95$

- Supply capacity for each supplier:
  - $u_{\text{Supplier1}} = 200$
  - $u_{\text{Supplier2}} = 250$
  - $u_{\text{Supplier3}} = 230$
  - $u_{\text{Supplier4}} = 220$
  - $u_{\text{Supplier5}} = 210$

- Transportation cost per unit from each supplier to each customer ($c_{s,c}$):

|              | Customer1 | Customer2 | Customer3 | Customer4 | Customer5 | Customer6 |
|--------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Supplier1    |     2     |     3     |     1     |     2     |     3     |     2     |
| Supplier2    |     1     |     2     |     3     |     2     |     3     |     2     |
| Supplier3    |     3     |     1     |     2     |     3     |     2     |     3     |
| Supplier4    |     2     |     3     |     2     |     1     |     3     |     4     |
| Supplier5    |     3     |     2     |     3     |     3     |     2     |     3     |

##### Decision Variables

- $x_{s,c} \geq 0$: Amount of produce shipped from supplier $s \in S$ to customer $c \in C$.

##### Objective Function

$\min \sum_{s \in S} \sum_{c \in C} c_{s,c} \cdot x_{s,c}$

##### Constraints

1. **Demand Satisfaction (for each customer):**

$\sum_{s \in S} x_{s,c} = d_c \quad \forall c \in C$

2. **Supply Capacity (for each supplier):**

$\sum_{c \in C} x_{s,c} \leq u_s \quad \forall s \in S$

3. **Non-negativity:**

$x_{s,c} \geq 0 \quad \forall s \in S, \forall c \in C$

##### Retrieved Information

{
  "customers": {
    "Customer1": 70,
    "Customer2": 80,
    "Customer3": 60,
    "Customer4": 90,
    "Customer5": 85,
    "Customer6": 95
  },
  "suppliers": {
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