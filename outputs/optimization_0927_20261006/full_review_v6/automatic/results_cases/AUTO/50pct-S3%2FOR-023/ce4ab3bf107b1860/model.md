##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier $i$ to customer (store) $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

##### Parameters

- Suppliers $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- Customers (Stores) $J = \{\text{Customer\_1}, \text{Customer\_2}, \text{Customer\_3}, \text{Customer\_4}, \text{Customer\_5}\}$
- Store-to-city mapping (from transportation_costs.csv columns): 
  - $J' = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$
- Demands:
  - $d_{\text{Customer\_1}} = 2397$
  - $d_{\text{Customer\_2}} = 1889$
  - $d_{\text{Customer\_3}} = 2518$
  - $d_{\text{Customer\_4}} = 3218$
  - $d_{\text{Customer\_5}} = 1813$
- Fixed costs:
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$
- Transportation costs $c_{ij}$ (supplier $i$, customer city $j'$):

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA         | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

- Let $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j' \in J'} c_{ij'} x_{ij'} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:** For each customer $j$ (store), the total quantity received from all suppliers must equal its demand.
   - $\sum_{i \in I} x_{ij'} = d_j, \quad \forall j \in J$
     - (Assuming mapping: Customer_1 $\rightarrow$ CLARINDA, Customer_2 $\rightarrow$ FORT MADISON, Customer_3 $\rightarrow$ SIOUX CITY, Customer_4 $\rightarrow$ TOLEDO, Customer_5 $\rightarrow$ BANCROFT)

2. **Supplier activation:** A supplier can only ship goods if it is activated.
   - $\sum_{j' \in J'} x_{ij'} \leq M y_i, \quad \forall i \in I$

3. **Variable domains:**
   - $x_{ij'} \geq 0$ (continuous), $\forall i \in I, j' \in J'$
   - $y_i \in \{0,1\}$, $\forall i \in I$

##### Full Model

Let $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$, $J' = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$, $J = \{\text{Customer\_1}, \text{Customer\_2}, \text{Customer\_3}, \text{Customer\_4}, \text{Customer\_5}\}$, and $M = 11835$.

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j' \in J'} c_{ij'} x_{ij'} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij'} = d_j, \quad \forall j \in J \\
& \sum_{j' \in J'} x_{ij'} \leq M y_i, \quad \forall i \in I \\
& x_{ij'} \geq 0, \quad \forall i \in I, j' \in J' \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

Where:

- $c_{ij'}$ is the transportation cost from supplier $i$ to customer city $j'$ as given in the table above.
- $f_i$ is the fixed cost for opening supplier $i$.
- $d_j$ is the demand for customer $j$ as given above.
- $x_{ij'}$ is the quantity shipped from supplier $i$ to customer city $j'$.
- $y_i$ is the binary activation variable for supplier $i$.

###### Retrieved Information

```json
{
  "suppliers": [
    "MOUNT AYR",
    "WAUKEE",
    "WAVERLY",
    "PELLA",
    "DES MOINES"
  ],
  "customers": [
    "Customer_1",
    "Customer_2",
    "Customer_3",
    "Customer_4",
    "Customer_5"
  ],
  "customer_cities": [
    "CLARINDA",
    "FORT MADISON",
    "SIOUX CITY",
    "TOLEDO",
    "BANCROFT"
  ],
  "demand": {
    "Customer_1": 2397,
    "Customer_2": 1889,
    "Customer_3": 2518,
    "Customer_4": 3218,
    "Customer_5": 1813
  },
  "fixed_cost": {
    "MOUNT AYR": 96.58,
    "WAUKEE": 94.06,
    "WAVERLY": 94.37,
    "PELLA": 82.88,
    "DES MOINES": 94.96
  },
  "cost": {
    "MOUNT AYR": {
      "CLARINDA": 694.68,
      "FORT MADISON": 17.48,
      "SIOUX CITY": 20.07,
      "TOLEDO": 199.02,
      "BANCROFT": 1685.53
    },
    "WAUKEE": {
      "CLARINDA": 15.13,
      "FORT MADISON": 1.5,
      "SIOUX CITY": 1.43,
      "TOLEDO": 27.88,
      "BANCROFT": 90.69
    },
    "WAVERLY": {
      "CLARINDA": 2.34,
      "FORT MADISON": 349.34,
      "SIOUX CITY": 246.6,
      "TOLEDO": 41.3,
      "BANCROFT": 78.73
    },
    "PELLA": {
      "CLARINDA": 1181.6,
      "FORT MADISON": 1458.53,
      "SIOUX CITY": 1646.36,
      "TOLEDO": 1924.55,
      "BANCROFT": 38.93
    },
    "DES MOINES": {
      "CLARINDA": 1030.8,
      "FORT MADISON": 43.48,
      "SIOUX CITY": 932.43,
      "TOLEDO": 55.39,
      "BANCROFT": 103.84
    }
  },
  "M": 11835
}
```