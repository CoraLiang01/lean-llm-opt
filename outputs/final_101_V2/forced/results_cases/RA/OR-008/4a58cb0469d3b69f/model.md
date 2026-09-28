Let $x_{ij}$ denote the amount of fresh produce shipped from Supplier $i$ to Customer $j$.

#### Sets and Indices
- Suppliers: $i \in \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$
- Customers: $j \in \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$

#### Parameters
- Demand for each customer:
  - $\text{Customer1}: 70$
  - $\text{Customer2}: 80$
  - $\text{Customer3}: 60$
  - $\text{Customer4}: 90$
  - $\text{Customer5}: 85$
  - $\text{Customer6}: 95$
- Supply capacity for each supplier:
  - $\text{Supplier1}: 200$
  - $\text{Supplier2}: 250$
  - $\text{Supplier3}: 230$
  - $\text{Supplier4}: 220$
  - $\text{Supplier5}: 210$
- Transportation costs per unit ($c_{ij}$):

|            | Customer1 | Customer2 | Customer3 | Customer4 | Customer5 | Customer6 |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Supplier1  |     2     |     3     |     1     |     2     |     3     |     2     |
| Supplier2  |     1     |     2     |     3     |     2     |     3     |     2     |
| Supplier3  |     3     |     1     |     2     |     3     |     2     |     3     |
| Supplier4  |     2     |     3     |     2     |     1     |     3     |     4     |
| Supplier5  |     3     |     2     |     3     |     3     |     2     |     3     |

#### Decision Variables
- $x_{ij} \geq 0$ (continuous), for all suppliers $i$ and customers $j$

#### Objective Function
Minimize total transportation cost:
$$
\min \sum_{i \in \text{Suppliers}} \sum_{j \in \text{Customers}} c_{ij} x_{ij}
$$

#### Constraints

1. **Demand Satisfaction (for each customer):**
   $$
   \sum_{i \in \text{Suppliers}} x_{ij} = d_j, \quad \forall j \in \text{Customers}
   $$
   where $d_j$ is the demand for customer $j$.

   Explicitly:
   - $\sum_{i} x_{i,\text{Customer1}} = 70$
   - $\sum_{i} x_{i,\text{Customer2}} = 80$
   - $\sum_{i} x_{i,\text{Customer3}} = 60$
   - $\sum_{i} x_{i,\text{Customer4}} = 90$
   - $\sum_{i} x_{i,\text{Customer5}} = 85$
   - $\sum_{i} x_{i,\text{Customer6}} = 95$

2. **Supply Capacity (for each supplier):**
   $$
   \sum_{j \in \text{Customers}} x_{ij} \leq s_i, \quad \forall i \in \text{Suppliers}
   $$
   where $s_i$ is the supply capacity for supplier $i$.

   Explicitly:
   - $\sum_{j} x_{\text{Supplier1},j} \leq 200$
   - $\sum_{j} x_{\text{Supplier2},j} \leq 250$
   - $\sum_{j} x_{\text{Supplier3},j} \leq 230$
   - $\sum_{j} x_{\text{Supplier4},j} \leq 220$
   - $\sum_{j} x_{\text{Supplier5},j} \leq 210$

3. **Non-negativity:**
   $$
   x_{ij} \geq 0, \quad \forall i, j
   $$

#### Complete Model

$$
\begin{align*}
\min \quad & \sum_{i \in \text{Suppliers}} \sum_{j \in \text{Customers}} c_{ij} x_{ij} \\
\text{s.t.} \quad & \sum_{i \in \text{Suppliers}} x_{ij} = d_j, \quad \forall j \in \text{Customers} \\
& \sum_{j \in \text{Customers}} x_{ij} \leq s_i, \quad \forall i \in \text{Suppliers} \\
& x_{ij} \geq 0, \quad \forall i, j
\end{align*}
$$

Where all parameters and indices are as listed above, and all data is used exactly as retrieved.