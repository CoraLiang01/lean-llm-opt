Let:
- S = {S1, S2, S3, S4, S5, S6, S7, S8, S9, S10} be the set of warehouses.
- C = {C1, C2, C3, C4, C5, C6, C7, C8, C9, C10} be the set of customers (stores).
- x_{i,j} = quantity shipped from warehouse i ∈ S to customer j ∈ C (continuous, x_{i,j} ≥ 0).

Parameters:
- demand_j: daily demand for customer j.
- supply_capacity_i: daily supply capacity for warehouse i.
- cost_{i,j}: transportation cost per unit from warehouse i to customer j.

Data (preserving source order):

Customer demands (from customer_demand.csv):
- demand_C1 = 45
- demand_C2 = 23
- demand_C3 = 94
- demand_C4 = 92
- demand_C5 = 57
- demand_C6 = 52
- demand_C7 = 23
- demand_C8 = 99
- demand_C9 = 99
- demand_C10 = 77

Warehouse supply capacities (from supply_capacity.csv):
- supply_capacity_S1 = 127
- supply_capacity_S2 = 236
- supply_capacity_S3 = 168
- supply_capacity_S4 = 115
- supply_capacity_S5 = 280
- supply_capacity_S6 = 179
- supply_capacity_S7 = 135
- supply_capacity_S8 = 263
- supply_capacity_S9 = 283
- supply_capacity_S10 = 476

Transportation costs (from transportation_costs.csv):

|        |   C1         |   C2         |   C3         |   C4         |   C5         |   C6         |   C7         |   C8         |   C9         |   C10        |
|--------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| S1     | 2077.0586725 | 0.0          | 54.33526480  | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 169.33026927 | 0.0          |
| S2     | 2077.0586725 | 0.0          | 1141.0405609 | 0.0          | 0.0          | 651.11123325 | 0.0          | 0.0          | 8.06334616   | 0.0          |
| S3     | 79.92102960  | 474.24509131 | 1477.0676289 | 22.58309959  | 474.24509131 | 41.10659696  | 474.24509131 | 474.24509131 | 624.16253950 | 474.24509131 |
| S4     | 1659.3369291 | 57.20541469  | 186.15190481 | 1201.3137084 | 1029.6974644 | 41.82210594  | 57.20541469  | 1201.3137084 | 884.56338707 | 1029.6974644 |
| S5     | 1297.2567041 | 77.76629131  | 24.26760228  | 1399.7932436 | 77.76629131  | 53.91161728  | 1399.7932436 | 77.76629131  | 1255.1151480 | 1399.7932436 |
| S6     | 1998.9090659 | 985.31654357 | 2.85416869   | 1149.5359675 | 985.31654357 | 730.69236477 | 54.73980798  | 985.31654357 | 46.80310221  | 1149.5359675 |
| S7     | 1780.3360050 | 0.0          | 1141.0405609 | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 8.06334616   | 0.0          |
| S8     | 75.40935896  | 1338.1987291 | 21.39134599  | 74.34437384  | 74.34437384  | 937.35062391 | 1338.1987291 | 1338.1987291 | 1392.1186581 | 1338.1987291 |
| S9     | 98.90755583  | 0.0          | 978.03476648 | 0.0          | 0.0          | 651.11123325 | 0.0          | 0.0          | 169.33026927 | 0.0          |
| S10    | 2077.0586725 | 0.0          | 54.33526480  | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 145.14023080 | 0.0          |

Mathematical Model:

Variables:
- For each i ∈ S, j ∈ C: x_{i,j} ≥ 0 (real, continuous)

Objective:
Minimize total transportation cost:
\[
\min \sum_{i \in S} \sum_{j \in C} cost_{i,j} \cdot x_{i,j}
\]
That is,
\[
\min \Bigg[
\begin{aligned}
&2077.0586725\,x_{S1,C1} + 0\,x_{S1,C2} + 54.33526480\,x_{S1,C3} + 0\,x_{S1,C4} + 0\,x_{S1,C5} + 36.17284629\,x_{S1,C6} + 0\,x_{S1,C7} + 0\,x_{S1,C8} + 169.33026927\,x_{S1,C9} + 0\,x_{S1,C10} \\
+&2077.0586725\,x_{S2,C1} + 0\,x_{S2,C2} + 1141.0405609\,x_{S2,C3} + 0\,x_{S2,C4} + 0\,x_{S2,C5} + 651.11123325\,x_{S2,C6} + 0\,x_{S2,C7} + 0\,x_{S2,C8} + 8.06334616\,x_{S2,C9} + 0\,x_{S2,C10} \\
+&79.92102960\,x_{S3,C1} + 474.24509131\,x_{S3,C2} + 1477.0676289\,x_{S3,C3} + 22.58309959\,x_{S3,C4} + 474.24509131\,x_{S3,C5} + 41.10659696\,x_{S3,C6} + 474.24509131\,x_{S3,C7} + 474.24509131\,x_{S3,C8} + 624.16253950\,x_{S3,C9} + 474.24509131\,x_{S3,C10} \\
+&1659.3369291\,x_{S4,C1} + 57.20541469\,x_{S4,C2} + 186.15190481\,x_{S4,C3} + 1201.3137084\,x_{S4,C4} + 1029.6974644\,x_{S4,C5} + 41.82210594\,x_{S4,C6} + 57.20541469\,x_{S4,C7} + 1201.3137084\,x_{S4,C8} + 884.56338707\,x_{S4,C9} + 1029.6974644\,x_{S4,C10} \\
+&1297.2567041\,x_{S5,C1} + 77.76629131\,x_{S5,C2} + 24.26760228\,x_{S5,C3} + 1399.7932436\,x_{S5,C4} + 77.76629131\,x_{S5,C5} + 53.91161728\,x_{S5,C6} + 1399.7932436\,x_{S5,C7} + 77.76629131\,x_{S5,C8} + 1255.1151480\,x_{S5,C9} + 1399.7932436\,x_{S5,C10} \\
+&1998.9090659\,x_{S6,C1} + 985.31654357\,x_{S6,C2} + 2.85416869\,x_{S6,C3} + 1149.5359675\,x_{S6,C4} + 985.31654357\,x_{S6,C5} + 730.69236477\,x_{S6,C6} + 54.73980798\,x_{S6,C7} + 985.31654357\,x_{S6,C8} + 46.80310221\,x_{S6,C9} + 1149.5359675\,x_{S6,C10} \\
+&1780.3360050\,x_{S7,C1} + 0\,x_{S7,C2} + 1141.0405609\,x_{S7,C3} + 0\,x_{S7,C4} + 0\,x_{S7,C5} + 36.17284629\,x_{S7,C6} + 0\,x_{S7,C7} + 0\,x_{S7,C8} + 8.06334616\,x_{S7,C9} + 0\,x_{S7,C10} \\
+&75.40935896\,x_{S8,C1} + 1338.1987291\,x_{S8,C2} + 21.39134599\,x_{S8,C3} + 74.34437384\,x_{S8,C4} + 74.34437384\,x_{S8,C5} + 937.35062391\,x_{S8,C6} + 1338.1987291\,x_{S8,C7} + 1338.1987291\,x_{S8,C8} + 1392.1186581\,x_{S8,C9} + 1338.1987291\,x_{S8,C10} \\
+&98.90755583\,x_{S9,C1} + 0\,x_{S9,C2} + 978.03476648\,x_{S9,C3} + 0\,x_{S9,C4} + 0\,x_{S9,C5} + 651.11123325\,x_{S9,C6} + 0\,x_{S9,C7} + 0\,x_{S9,C8} + 169.33026927\,x_{S9,C9} + 0\,x_{S9,C10} \\
+&2077.0586725\,x_{S10,C1} + 0\,x_{S10,C2} + 54.33526480\,x_{S10,C3} + 0\,x_{S10,C4} + 0\,x_{S10,C5} + 36.17284629\,x_{S10,C6} + 0\,x_{S10,C7} + 0\,x_{S10,C8} + 145.14023080\,x_{S10,C9} + 0\,x_{S10,C10}
\end{aligned}
\Bigg]
\]

Subject to:

1. Demand satisfaction for each customer j ∈ C:
\[
\sum_{i \in S} x_{i,j} = demand_j
\]
Explicitly:
\[
\begin{aligned}
&x_{S1,C1} + x_{S2,C1} + x_{S3,C1} + x_{S4,C1} + x_{S5,C1} + x_{S6,C1} + x_{S7,C1} + x_{S8,C1} + x_{S9,C1} + x_{S10,C1} = 45 \\
&x_{S1,C2} + x_{S2,C2} + x_{S3,C2} + x_{S4,C2} + x_{S5,C2} + x_{S6,C2} + x_{S7,C2} + x_{S8,C2} + x_{S9,C2} + x_{S10,C2} = 23 \\
&x_{S1,C3} + x_{S2,C3} + x_{S3,C3} + x_{S4,C3} + x_{S5,C3} + x_{S6,C3} + x_{S7,C3} + x_{S8,C3} + x_{S9,C3} + x_{S10,C3} = 94 \\
&x_{S1,C4} + x_{S2,C4} + x_{S3,C4} + x_{S4,C4} + x_{S5,C4} + x_{S6,C4} + x_{S7,C4} + x_{S8,C4} + x_{S9,C4} + x_{S10,C4} = 92 \\
&x_{S1,C5} + x_{S2,C5} + x_{S3,C5} + x_{S4,C5} + x_{S5,C5} + x_{S6,C5} + x_{S7,C5} + x_{S8,C5} + x_{S9,C5} + x_{S10,C5} = 57 \\
&x_{S1,C6} + x_{S2,C6} + x_{S3,C6} + x_{S4,C6} + x_{S5,C6} + x_{S6,C6} + x_{S7,C6} + x_{S8,C6} + x_{S9,C6} + x_{S10,C6} = 52 \\
&x_{S1,C7} + x_{S2,C7} + x_{S3,C7} + x_{S4,C7} + x_{S5,C7} + x_{S6,C7} + x_{S7,C7} + x_{S8,C7} + x_{S9,C7} + x_{S10,C7} = 23 \\
&x_{S1,C8} + x_{S2,C8} + x_{S3,C8} + x_{S4,C8} + x_{S5,C8} + x_{S6,C8} + x_{S7,C8} + x_{S8,C8} + x_{S9,C8} + x_{S10,C8} = 99 \\
&x_{S1,C9} + x_{S2,C9} + x_{S3,C9} + x_{S4,C9} + x_{S5,C9} + x_{S6,C9} + x_{S7,C9} + x_{S8,C9} + x_{S9,C9} + x_{S10,C9} = 99 \\
&x_{S1,C10} + x_{S2,C10} + x_{S3,C10} + x_{S4,C10} + x_{S5,C10} + x_{S6,C10} + x_{S7,C10} + x_{S8,C10} + x_{S9,C10} + x_{S10,C10} = 77 \\
\end{aligned}
\]

2. Supply capacity for each warehouse i ∈ S:
\[
\sum_{j \in C} x_{i,j} \leq supply\_capacity_i
\]
Explicitly:
\[
\begin{aligned}
&x_{S1,C1} + x_{S1,C2} + x_{S1,C3} + x_{S1,C4} + x_{S1,C5} + x_{S1,C6} + x_{S1,C7} + x_{S1,C8} + x_{S1,C9} + x_{S1,C10} \leq 127 \\
&x_{S2,C1} + x_{S2,C2} + x_{S2,C3} + x_{S2,C4} + x_{S2,C5} + x_{S2,C6} + x_{S2,C7} + x_{S2,C8} + x_{S2,C9} + x_{S2,C10} \leq 236 \\
&x_{S3,C1} + x_{S3,C2} + x_{S3,C3} + x_{S3,C4} + x_{S3,C5} + x_{S3,C6} + x_{S3,C7} + x_{S3,C8} + x_{S3,C9} + x_{S3,C10} \leq 168 \\
&x_{S4,C1} + x_{S4,C2} + x_{S4,C3} + x_{S4,C4} + x_{S4,C5} + x_{S4,C6} + x_{S4,C7} + x_{S4,C8} + x_{S4,C9} + x_{S4,C10} \leq 115 \\
&x_{S5,C1} + x_{S5,C2} + x_{S5,C3} + x_{S5,C4} + x_{S5,C5} + x_{S5,C6} + x_{S5,C7} + x_{S5,C8} + x_{S5,C9} + x_{S5,C10} \leq 280 \\
&x_{S6,C1} + x_{S6,C2} + x_{S6,C3} + x_{S6,C4} + x_{S6,C5} + x_{S6,C6} + x_{S6,C7} + x_{S6,C8} + x_{S6,C9} + x_{S6,C10} \leq 179 \\
&x_{S7,C1} + x_{S7,C2} + x_{S7,C3} + x_{S7,C4} + x_{S7,C5} + x_{S7,C6} + x_{S7,C7} + x_{S7,C8} + x_{S7,C9} + x_{S7,C10} \leq 135 \\
&x_{S8,C1} + x_{S8,C2} + x_{S8,C3} + x_{S8,C4} + x_{S8,C5} + x_{S8,C6} + x_{S8,C7} + x_{S8,C8} + x_{S8,C9} + x_{S8,C10} \leq 263 \\
&x_{S9,C1} + x_{S9,C2} + x_{S9,C3} + x_{S9,C4} + x_{S9,C5} + x_{S9,C6} + x_{S9,C7} + x_{S9,C8} + x_{S9,C9} + x_{S9,C10} \leq 283 \\
&x_{S10,C1} + x_{S10,C2} + x_{S10,C3} + x_{S10,C4} + x_{S10,C5} + x_{S10,C6} + x_{S10,C7} + x_{S10,C8} + x_{S10,C9} + x_{S10,C10} \leq 476 \\
\end{aligned}
\]

3. Non-negativity:
\[
x_{i,j} \geq 0 \quad \forall i \in S,\, j \in C
\]

This is a complete numerical linear programming formulation for the transportation problem as described, using all coefficients and identifiers from the provided CSVs, preserving source order and semantics.