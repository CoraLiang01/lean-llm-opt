Let:
- $y_i \in \{0,1\}$: 1 if facility $i$ is constructed, 0 otherwise, for each facility $i \in \{\text{A1},\ldots,\text{A15}\}$.
- $x_{ij} \geq 0$: amount shipped from facility $i$ to distribution center $j$, for each $i \in \{\text{A1},\ldots,\text{A15}\}$ and $j \in \{\text{B1},\ldots,\text{B8}\}$.

Parameters (from facility_costs.csv, shipping_costs.csv, demand_requirements.csv):

Facility fixed costs and capacities:
\[
\begin{array}{lll}
\text{Facility} & \text{FixedCost} & \text{Capacity} \\
\text{A1} & 0 & 30 \\
\text{A2} & 175 & 10 \\
\text{A3} & 300 & 20 \\
\text{A4} & 375 & 30 \\
\text{A5} & 500 & 40 \\
\text{A6} & 200 & 20 \\
\text{A7} & 260 & 25 \\
\text{A8} & 220 & 30 \\
\text{A9} & 320 & 35 \\
\text{A10} & 280 & 20 \\
\text{A11} & 350 & 40 \\
\text{A12} & 420 & 25 \\
\text{A13} & 470 & 30 \\
\text{A14} & 520 & 50 \\
\text{A15} & 560 & 45 \\
\end{array}
\]

Shipping costs (per unit shipped from $i$ to $j$):

\[
\begin{array}{l|cccccccc}
 & \text{B1} & \text{B2} & \text{B3} & \text{B4} & \text{B5} & \text{B6} & \text{B7} & \text{B8} \\
\hline
\text{A1} & 8 & 4 & 3 & 6 & 7 & 5 & 9 & 8 \\
\text{A2} & 5 & 2 & 3 & 5 & 6 & 4 & 7 & 6 \\
\text{A3} & 4 & 3 & 4 & 6 & 5 & 5 & 6 & 7 \\
\text{A4} & 9 & 7 & 5 & 8 & 9 & 6 & 10 & 7 \\
\text{A5} & 10 & 4 & 2 & 6 & 8 & 5 & 7 & 3 \\
\text{A6} & 6 & 5 & 4 & 5 & 7 & 6 & 8 & 5 \\
\text{A7} & 7 & 6 & 5 & 4 & 6 & 7 & 9 & 6 \\
\text{A8} & 5 & 4 & 6 & 3 & 5 & 6 & 7 & 6 \\
\text{A9} & 8 & 7 & 6 & 7 & 9 & 8 & 10 & 7 \\
\text{A10} & 6 & 5 & 7 & 4 & 6 & 5 & 7 & 5 \\
\text{A11} & 9 & 6 & 4 & 6 & 8 & 7 & 9 & 6 \\
\text{A12} & 7 & 5 & 6 & 5 & 6 & 5 & 8 & 5 \\
\text{A13} & 8 & 6 & 5 & 6 & 7 & 6 & 8 & 7 \\
\text{A14} & 9 & 5 & 3 & 5 & 7 & 4 & 6 & 4 \\
\text{A15} & 10 & 6 & 4 & 5 & 8 & 5 & 7 & 5 \\
\end{array}
\]

Demand at each distribution center:
\[
\begin{array}{ll}
\text{B1}: & 30 \\
\text{B2}: & 25 \\
\text{B3}: & 20 \\
\text{B4}: & 35 \\
\text{B5}: & 25 \\
\text{B6}: & 30 \\
\text{B7}: & 25 \\
\text{B8}: & 30 \\
\end{array}
\]

---

#### Mathematical Model

Minimize total system cost:
\[
\min \left( \sum_{i=\text{A1}}^{\text{A15}} \text{FixedCost}_i \cdot y_i + \sum_{i=\text{A1}}^{\text{A15}} \sum_{j=\text{B1}}^{\text{B8}} \text{ShipCost}_{ij} \cdot x_{ij} \right)
\]

Subject to:

1. **Demand satisfaction at each distribution center:**
   \[
   \sum_{i=\text{A1}}^{\text{A15}} x_{ij} = \text{Demand}_j \qquad \forall j \in \{\text{B1},\ldots,\text{B8}\}
   \]

2. **Facility capacity (can only ship if open):**
   \[
   \sum_{j=\text{B1}}^{\text{B8}} x_{ij} \leq \text{Capacity}_i \cdot y_i \qquad \forall i \in \{\text{A1},\ldots,\text{A15}\}
   \]

3. **Variable domains:**
   \[
   y_i \in \{0,1\} \qquad \forall i \in \{\text{A1},\ldots,\text{A15}\}
   \]
   \[
   x_{ij} \geq 0 \qquad \forall i \in \{\text{A1},\ldots,\text{A15}\},\ j \in \{\text{B1},\ldots,\text{B8}\}
   \]

---

All coefficients and identifiers are as retrieved and preserved in source order.