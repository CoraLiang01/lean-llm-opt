##### Mathematical Optimization Model

Let $i$ index all products classified as ‘ELE-S’ (i.e., all products whose Product_Reference begins with ‘ELE-’), in the source order below. For each such product $i$, let:

- $A_i$ = Revenue of product $i$
- $d_i$ = Demand of product $i$
- $I_i$ = Initial Inventory of product $i$
- $x_i$ = number of units of product $i$ to fulfill (decision variable)

The model is:

$$
\begin{align*}
\text{Maximize} \quad & \sum_{i} A_i \cdot x_i \\
\text{subject to} \quad & 0 \leq x_i \leq \min\{d_i, I_i\} \quad \forall i \\
& x_i \in \mathbb{Z} \quad \forall i
\end{align*}
$$

##### Retrieved Information

Source-ordered list of all ‘ELE-S’ products with their parameters:

| Product_Reference      | Revenue | Demand | Initial Inventory |
|-----------------------|---------|--------|------------------|
| ELE-ACC-10000478      | 2.0     | 140    | 1000.0           |
| ELE-ACC-10002424      | 2.0     | 136    | 1000.0           |
| ELE-ACC-10018567      | 2.0     | 268    | 2000.0           |
| ELE-ACC-10019567      | 2.0     | 121    | 1000.0           |
| ELE-CAM-10000475      | 12.0    | 812    | 6000.0           |
| ELE-CAM-10002121      | 12.0    | 875    | 6000.0           |
| ELE-CAM-10015234      | 12.0    | 1519   | 12000.0          |
| ELE-CAM-10016234      | 12.0    | 868    | 6000.0           |
| ELE-HEA-10000460      | 3.0     | 192    | 1500.0           |
| ELE-HEA-10000493      | 3.6     | 230    | 1800.0           |
| ELE-HEA-10003939      | 3.6     | 264    | 1800.0           |
| ELE-HEA-10006666      | 3.0     | 195    | 1500.0           |
| ELE-HEA-10006789      | 3.0     | 612    | 4500.0           |
| ELE-HEA-10008901      | 3.0     | 203    | 1500.0           |
| ELE-HEA-10033012      | 3.6     | 223    | 1800.0           |
| ELE-HEA-10034234      | 3.6     | 266    | 1800.0           |
| ELE-LAP-10000457      | 16.0    | 982    | 8000.0           |
| ELE-LAP-10000490      | 17.0    | 1268   | 8500.0           |
| ELE-LAP-10003333      | 16.0    | 1175   | 8000.0           |
| ELE-LAP-10003456      | 16.0    | 3515   | 24000.0          |
| ELE-LAP-10003636      | 17.0    | 1113   | 8500.0           |
| ELE-LAP-10004567      | 16.0    | 1104   | 8000.0           |
| ELE-LAP-10030789      | 17.0    | 1153   | 8500.0           |
| ELE-LAP-10031901      | 17.0    | 1200   | 8500.0           |
| ELE-MON-10000472      | 7.0     | 453    | 3500.0           |
| ELE-MON-10001818      | 7.0     | 486    | 3500.0           |
| ELE-MON-10013901      | 7.0     | 1371   | 10500.0          |
| ELE-MON-10020543      | 7.0     | 455    | 3500.0           |
| ELE-PRI-10000466      | 5.6     | 366    | 2800.0           |
| ELE-PRI-10000481      | 6.0     | 382    | 3000.0           |
| ELE-PRI-10001212      | 5.6     | 412    | 2800.0           |
| ELE-PRI-10002727      | 6.0     | 381    | 3000.0           |
| ELE-PRI-10009345      | 5.6     | 1121   | 8400.0           |
| ELE-PRI-10014567      | 5.6     | 390    | 2800.0           |
| ELE-PRI-10021890      | 6.0     | 848    | 6000.0           |
| ELE-PRI-10022901      | 6.0     | 419    | 3000.0           |
| ELE-SMA-10000463      | 4.0     | 295    | 2000.0           |
| ELE-SMA-10000487      | 14.0    | 1002   | 7000.0           |
| ELE-SMA-10003333      | 14.0    | 958    | 7000.0           |
| ELE-SMA-10009012      | 4.0     | 777    | 6000.0           |
| ELE-SMA-10009999      | 4.0     | 271    | 2000.0           |
| ELE-SMA-10011234      | 4.0     | 244    | 2000.0           |
| ELE-SMA-10027456      | 14.0    | 990    | 7000.0           |
| ELE-SMA-10028567      | 14.0    | 1000   | 7000.0           |
| ELE-SPE-10000484      | 2.4     | 169    | 1200.0           |
| ELE-SPE-10003030      | 2.4     | 155    | 1200.0           |
| ELE-SPE-10024123      | 2.4     | 327    | 2400.0           |
| ELE-SPE-10025234      | 2.4     | 174    | 1200.0           |
| ELE-TAB-10000469      | 9.0     | 669    | 4500.0           |
| ELE-TAB-10001515      | 9.0     | 623    | 4500.0           |
| ELE-TAB-10010678      | 9.0     | 1716   | 13500.0          |
| ELE-TAB-10017689      | 9.0     | 564    | 4500.0           |

Where for each $i$, $x_i$ is the integer number of units to fulfill, with $0 \leq x_i \leq \min\{d_i, I_i\}$. The objective is to maximize total revenue.