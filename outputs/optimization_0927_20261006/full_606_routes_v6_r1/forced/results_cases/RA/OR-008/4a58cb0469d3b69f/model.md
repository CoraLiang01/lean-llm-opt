Let $x_{ij}$ denote the amount of fresh produce shipped from supplier $i$ to customer $j$.

**Sets and Indices:**
- Suppliers $i \in \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$
- Customers $j \in \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$

**Parameters:**
- $c_{ij}$: Transportation cost per unit from supplier $i$ to customer $j$ (see table below)
- $d_j$: Demand of customer $j$
- $s_i$: Supply capacity of supplier $i$

**Data:**

Customer Demands:
| Customer   | Demand ($d_j$) |
|------------|---------------|
| Customer1  | 70            |
| Customer2  | 80            |
| Customer3  | 60            |
| Customer4  | 90            |
| Customer5  | 85            |
| Customer6  | 95            |

Supplier Capacities:
| Supplier   | Supply Capacity ($s_i$) |
|------------|------------------------|
| Supplier1  | 200                    |
| Supplier2  | 250                    |
| Supplier3  | 230                    |
| Supplier4  | 220                    |
| Supplier5  | 210                    |

Transportation Costs ($c_{ij}$):

|            | Customer1 | Customer2 | Customer3 | Customer4 | Customer5 | Customer6 |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Supplier1  |     2     |     3     |     1     |     2     |     3     |     2     |
| Supplier2  |     1     |     2     |     3     |     2     |     3     |     2     |
| Supplier3  |     3     |     1     |     2     |     3     |     2     |     3     |
| Supplier4  |     2     |     3     |     2     |     1     |     3     |     4     |
| Supplier5  |     3     |     2     |     3     |     3     |     2     |     3     |

**Mathematical Model:**

Minimize total transportation cost:
$$
\min \sum_{i \in \text{Suppliers}} \sum_{j \in \text{Customers}} c_{ij} x_{ij}
$$

Subject to:

1. **Demand satisfaction for each customer:**
   $$
   \sum_{i \in \text{Suppliers}} x_{ij} = d_j, \quad \forall j \in \text{Customers}
   $$
   That is,
   \begin{align*}
   x_{\text{Supplier1},j} + x_{\text{Supplier2},j} + x_{\text{Supplier3},j} + x_{\text{Supplier4},j} + x_{\text{Supplier5},j} = d_j, \quad \forall j
   \end{align*}

2. **Supply capacity for each supplier:**
   $$
   \sum_{j \in \text{Customers}} x_{ij} \leq s_i, \quad \forall i \in \text{Suppliers}
   $$
   That is,
   \begin{align*}
   x_{i,\text{Customer1}} + x_{i,\text{Customer2}} + x_{i,\text{Customer3}} + x_{i,\text{Customer4}} + x_{i,\text{Customer5}} + x_{i,\text{Customer6}} \leq s_i, \quad \forall i
   \end{align*}

3. **Non-negativity:**
   $$
   x_{ij} \geq 0, \quad \forall i \in \text{Suppliers},\; j \in \text{Customers}
   $$

**Where:**
- $x_{ij}$: Amount of fresh produce shipped from supplier $i$ to customer $j$ (continuous, $\geq 0$)
- $c_{ij}$: Transportation cost per unit as given in the table above
- $d_j$: Demand for customer $j$ as given above
- $s_i$: Supply capacity for supplier $i$ as given above

**All data and identifiers are preserved in source order as retrieved.**