##### Sets and Indices

- Suppliers: $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}\}$
- Customers (Stores): $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}\}$

##### Parameters

- Fixed costs for opening supplier $i$: $f_i$
- Transportation cost per unit from supplier $i$ to customer $j$: $c_{ij}$
- Demand at customer $j$: $d_j$

###### Fixed Costs

\[
\begin{align*}
f_{\text{S1}} &= 98.88 \\
f_{\text{S2}} &= 99.73 \\
f_{\text{S3}} &= 94.01 \\
f_{\text{S4}} &= 93.77 \\
f_{\text{S5}} &= 107.59 \\
f_{\text{S6}} &= 112.65 \\
\end{align*}
\]

###### Transportation Costs

\[
\begin{array}{c|cccccc}
 & \text{C1} & \text{C2} & \text{C3} & \text{C4} & \text{C5} & \text{C6} \\
\hline
\text{S1} & 0.08 & 52.33 & 73.57 & 1237.33 & 0.07 & 112.16 \\
\text{S2} & 46.02 & 175.23 & 2026.83 & 299.89 & 966.53 & 1590.42 \\
\text{S3} & 1031.74 & 78.13 & 99.02 & 277.07 & 884.45 & 1800.86 \\
\text{S4} & 868.75 & 94.2 & 1776.34 & 285.48 & 868.85 & 86.55 \\
\text{S5} & 1577 & 760.15 & 2090.19 & 43.2 & 1577.12 & 1095.17 \\
\text{S6} & 49.14 & 4.33 & 2079.57 & 277.04 & 1032.01 & 1543.49 \\
\end{array}
\]

###### Demands

\[
\begin{align*}
d_{\text{C1}} &= 216 \\
d_{\text{C2}} &= 216 \\
d_{\text{C3}} &= 216 \\
d_{\text{C4}} &= 144 \\
d_{\text{C5}} &= 144 \\
d_{\text{C6}} &= 144 \\
\end{align*}
\]

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise.
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$.

##### Objective Function

\[
\min \left( \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \right)
\]

##### Constraints

1. **Demand Satisfaction:**  
   For each customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Supplier Activation:**  
   For each supplier $i \in I$ and customer $j \in J$,
   \[
   x_{ij} \leq d_j y_i
   \]
   (A supplier can only ship to a customer if it is open; $d_j$ is an upper bound since no customer can receive more than its demand.)

3. **Variable Domains:**  
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Retrieved Information

{
  "suppliers": ["S1", "S2", "S3", "S4", "S5", "S6"],
  "customers": ["C1", "C2", "C3", "C4", "C5", "C6"],
  "fixed_costs": {
    "S1": 98.88,
    "S2": 99.73,
    "S3": 94.01,
    "S4": 93.77,
    "S5": 107.59,
    "S6": 112.65
  },
  "transportation_costs": {
    "S1": {"C1": 0.08, "C2": 52.33, "C3": 73.57, "C4": 1237.33, "C5": 0.07, "C6": 112.16},
    "S2": {"C1": 46.02, "C2": 175.23, "C3": 2026.83, "C4": 299.89, "C5": 966.53, "C6": 1590.42},
    "S3": {"C1": 1031.74, "C2": 78.13, "C3": 99.02, "C4": 277.07, "C5": 884.45, "C6": 1800.86},
    "S4": {"C1": 868.75, "C2": 94.2, "C3": 1776.34, "C4": 285.48, "C5": 868.85, "C6": 86.55},
    "S5": {"C1": 1577, "C2": 760.15, "C3": 2090.19, "C4": 43.2, "C5": 1577.12, "C6": 1095.17},
    "S6": {"C1": 49.14, "C2": 4.33, "C3": 2079.57, "C4": 277.04, "C5": 1032.01, "C6": 1543.49}
  },
  "demands": {
    "C1": 216,
    "C2": 216,
    "C3": 216,
    "C4": 144,
    "C5": 144,
    "C6": 144
  }
}