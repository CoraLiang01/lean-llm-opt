##### Sets

- $I = \{\text{Facility\_1}, \text{Facility\_2}, \text{Facility\_3}, \text{Facility\_4}, \text{Facility\_5}\}$: set of suppliers (facilities)
- $J = \{\text{Customer\_1}, \text{Customer\_2}, \text{Customer\_3}, \text{Customer\_4}, \text{Customer\_5}\}$: set of stores (customers)

##### Parameters

- Fixed costs for each supplier:
  - $f_{\text{Facility\_1}} = 96.58$
  - $f_{\text{Facility\_2}} = 94.06$
  - $f_{\text{Facility\_3}} = 94.37$
  - $f_{\text{Facility\_4}} = 82.88$
  - $f_{\text{Facility\_5}} = 94.96$

- Demand for each store:
  - $d_{\text{Customer\_1}} = 2397$
  - $d_{\text{Customer\_2}} = 1889$
  - $d_{\text{Customer\_3}} = 2518$
  - $d_{\text{Customer\_4}} = 3218$
  - $d_{\text{Customer\_5}} = 1813$

- Transportation cost per unit from each supplier to each store:

\[
\begin{array}{c|ccccc}
 & \text{Customer\_1} & \text{Customer\_2} & \text{Customer\_3} & \text{Customer\_4} & \text{Customer\_5} \\
\hline
\text{Facility\_1} & 694.68 & 17.48 & 20.07 & 199.02 & 1685.53 \\
\text{Facility\_2} & 15.13 & 1.50 & 1.43 & 27.88 & 90.69 \\
\text{Facility\_3} & 2.34 & 349.34 & 246.60 & 41.30 & 78.73 \\
\text{Facility\_4} & 1181.60 & 1458.53 & 1646.36 & 1924.55 & 38.93 \\
\text{Facility\_5} & 1030.80 & 43.48 & 932.43 & 55.39 & 103.84 \\
\end{array}
\]

Let $c_{ij}$ denote the transportation cost per unit from supplier $i$ to store $j$ as given above.

##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to store $j \in J$ (continuous)
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:** Each store must receive exactly its demand.
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation:** No shipments from inactive suppliers.
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Complete Model

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

where

- $I = \{\text{Facility\_1}, \text{Facility\_2}, \text{Facility\_3}, \text{Facility\_4}, \text{Facility\_5}\}$
- $J = \{\text{Customer\_1}, \text{Customer\_2}, \text{Customer\_3}, \text{Customer\_4}, \text{Customer\_5}\}$
- $f_i$, $d_j$, $c_{ij}$ as specified above
- $M = 11835$