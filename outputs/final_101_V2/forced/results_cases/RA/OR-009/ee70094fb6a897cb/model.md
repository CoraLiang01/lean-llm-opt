Let $x_{ij}$ denote the quantity of beverages shipped from plant $i$ to customer $j$.

#### Sets and Parameters (from retrieved data):

Plants (from supply_capacity.csv and transportation_costs.csv Unnamed: 0):  
$S = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$

Customers (from customer_demand.csv and transportation_costs.csv columns):  
$C = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$

Supply capacities:
\[
\begin{align*}
\text{S1}: &\quad 2531 \\
\text{S2}: &\quad 20 \\
\text{S3}: &\quad 210 \\
\text{S4}: &\quad 241 \\
\end{align*}
\]

Customer demands:
\[
\begin{align*}
\text{C1}: &\quad 94 \\
\text{C2}: &\quad 39 \\
\text{C3}: &\quad 65 \\
\text{C4}: &\quad 435 \\
\end{align*}
\]

Transportation costs per unit (from transportation_costs.csv):

\[
\begin{array}{c|cccc}
 & \text{C1} & \text{C2} & \text{C3} & \text{C4} \\
\hline
\text{S1} & 543.756480860856 & 23.685276141764653 & 23.676386730773032 & 447.75143678673766 \\
\text{S2} & 883.9151090405642 & 0.04977684765576961 & 0.0350986687216299 & 44.45588531711622 \\
\text{S3} & 537.3456896658107 & 23.769274659075112 & 498.95659249465467 & 440.60737890439776 \\
\text{S4} & 1791.493192397229 & 68.21633865655126 & 1432.4837339656747 & 1527.7635425462734 \\
\end{array}
\]

#### Decision Variables

For all $i \in S$, $j \in C$:
\[
x_{ij} \geq 0 \quad \text{(continuous, quantity of beverages shipped from plant $i$ to customer $j$)}
\]

#### Objective Function

Minimize total transportation cost:
\[
\min \sum_{i \in S} \sum_{j \in C} c_{ij} x_{ij}
\]
where $c_{ij}$ is the transportation cost per unit from plant $i$ to customer $j$ (see table above).

#### Constraints

1. **Supply capacity at each plant:**
\[
\sum_{j \in C} x_{ij} \leq \text{supply\_capacity}_i \qquad \forall i \in S
\]
Numerically:
\[
\begin{align*}
x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} + x_{\text{S1},\text{C3}} + x_{\text{S1},\text{C4}} &\leq 2531 \\
x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} + x_{\text{S2},\text{C3}} + x_{\text{S2},\text{C4}} &\leq 20 \\
x_{\text{S3},\text{C1}} + x_{\text{S3},\text{C2}} + x_{\text{S3},\text{C3}} + x_{\text{S3},\text{C4}} &\leq 210 \\
x_{\text{S4},\text{C1}} + x_{\text{S4},\text{C2}} + x_{\text{S4},\text{C3}} + x_{\text{S4},\text{C4}} &\leq 241 \\
\end{align*}
\]

2. **Demand satisfaction at each customer:**
\[
\sum_{i \in S} x_{ij} \geq \text{demand}_j \qquad \forall j \in C
\]
Numerically:
\[
\begin{align*}
x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + x_{\text{S3},\text{C1}} + x_{\text{S4},\text{C1}} &\geq 94 \\
x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + x_{\text{S3},\text{C2}} + x_{\text{S4},\text{C2}} &\geq 39 \\
x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + x_{\text{S3},\text{C3}} + x_{\text{S4},\text{C3}} &\geq 65 \\
x_{\text{S1},\text{C4}} + x_{\text{S2},\text{C4}} + x_{\text{S3},\text{C4}} + x_{\text{S4},\text{C4}} &\geq 435 \\
\end{align*}
\]

3. **Nonnegativity:**
\[
x_{ij} \geq 0 \qquad \forall i \in S,\, j \in C
\]

---

#### Complete Model

Minimize
\[
543.756480860856\, x_{\text{S1},\text{C1}} + 23.685276141764653\, x_{\text{S1},\text{C2}} + 23.676386730773032\, x_{\text{S1},\text{C3}} + 447.75143678673766\, x_{\text{S1},\text{C4}}
\]
\[
+ 883.9151090405642\, x_{\text{S2},\text{C1}} + 0.04977684765576961\, x_{\text{S2},\text{C2}} + 0.0350986687216299\, x_{\text{S2},\text{C3}} + 44.45588531711622\, x_{\text{S2},\text{C4}}
\]
\[
+ 537.3456896658107\, x_{\text{S3},\text{C1}} + 23.769274659075112\, x_{\text{S3},\text{C2}} + 498.95659249465467\, x_{\text{S3},\text{C3}} + 440.60737890439776\, x_{\text{S3},\text{C4}}
\]
\[
+ 1791.493192397229\, x_{\text{S4},\text{C1}} + 68.21633865655126\, x_{\text{S4},\text{C2}} + 1432.4837339656747\, x_{\text{S4},\text{C3}} + 1527.7635425462734\, x_{\text{S4},\text{C4}}
\]

Subject to:
\[
\begin{align*}
x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} + x_{\text{S1},\text{C3}} + x_{\text{S1},\text{C4}} &\leq 2531 \\
x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} + x_{\text{S2},\text{C3}} + x_{\text{S2},\text{C4}} &\leq 20 \\
x_{\text{S3},\text{C1}} + x_{\text{S3},\text{C2}} + x_{\text{S3},\text{C3}} + x_{\text{S3},\text{C4}} &\leq 210 \\
x_{\text{S4},\text{C1}} + x_{\text{S4},\text{C2}} + x_{\text{S4},\text{C3}} + x_{\text{S4},\text{C4}} &\leq 241 \\
x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + x_{\text{S3},\text{C1}} + x_{\text{S4},\text{C1}} &\geq 94 \\
x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + x_{\text{S3},\text{C2}} + x_{\text{S4},\text{C2}} &\geq 39 \\
x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + x_{\text{S3},\text{C3}} + x_{\text{S4},\text{C3}} &\geq 65 \\
x_{\text{S1},\text{C4}} + x_{\text{S2},\text{C4}} + x_{\text{S3},\text{C4}} + x_{\text{S4},\text{C4}} &\geq 435 \\
x_{ij} &\geq 0 \qquad \forall i \in S,\, j \in C \\
\end{align*}
\]

All variables $x_{ij}$ are continuous and nonnegative.