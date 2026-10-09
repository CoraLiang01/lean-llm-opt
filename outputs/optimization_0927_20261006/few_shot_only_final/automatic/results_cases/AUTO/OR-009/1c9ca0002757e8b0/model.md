**Sets:**  
Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (plants)  
Let $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets)

**Parameters:**  
Demands:  
$d_{\text{C1}} = 94$  
$d_{\text{C2}} = 39$  
$d_{\text{C3}} = 65$  
$d_{\text{C4}} = 435$  

Supply capacities:  
$s_{\text{S1}} = 2531$  
$s_{\text{S2}} = 20$  
$s_{\text{S3}} = 210$  
$s_{\text{S4}} = 241$  

Transportation costs $c_{ij}$:  

|        | C1                | C2                | C3                | C4                |
|--------|-------------------|-------------------|-------------------|-------------------|
| S1     | 543.756480860856  | 23.685276141764653| 23.676386730773032| 447.75143678673766|
| S2     | 883.9151090405642 | 0.04977684765576961| 0.0350986687216299| 44.45588531711622 |
| S3     | 537.3456896658107 | 23.769274659075112| 498.95659249465467| 440.60737890439776|
| S4     | 1791.493192397229 | 68.21633865655126 | 1432.4837339656747| 1527.7635425462734|

**Decision Variables:**  
For all $i \in I$, $j \in J$:  
$x_{ij} \geq 0$ = quantity shipped from plant $i$ to outlet $j$ (continuous)

**Objective:**  
Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$
That is,
\[
\min \Bigg[
\begin{aligned}
&543.756480860856\, x_{\text{S1},\text{C1}} + 23.685276141764653\, x_{\text{S1},\text{C2}} + 23.676386730773032\, x_{\text{S1},\text{C3}} + 447.75143678673766\, x_{\text{S1},\text{C4}} \\
+&883.9151090405642\, x_{\text{S2},\text{C1}} + 0.04977684765576961\, x_{\text{S2},\text{C2}} + 0.0350986687216299\, x_{\text{S2},\text{C3}} + 44.45588531711622\, x_{\text{S2},\text{C4}} \\
+&537.3456896658107\, x_{\text{S3},\text{C1}} + 23.769274659075112\, x_{\text{S3},\text{C2}} + 498.95659249465467\, x_{\text{S3},\text{C3}} + 440.60737890439776\, x_{\text{S3},\text{C4}} \\
+&1791.493192397229\, x_{\text{S4},\text{C1}} + 68.21633865655126\, x_{\text{S4},\text{C2}} + 1432.4837339656747\, x_{\text{S4},\text{C3}} + 1527.7635425462734\, x_{\text{S4},\text{C4}}
\end{aligned}
\Bigg]
\]

**Subject to:**

1. **Demand satisfaction (each outlet receives at least its demand):**
   - $x_{\text{S1},j} + x_{\text{S2},j} + x_{\text{S3},j} + x_{\text{S4},j} \geq d_j$ for all $j \in J$
   - Explicitly:
     - $x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + x_{\text{S3},\text{C1}} + x_{\text{S4},\text{C1}} \geq 94$
     - $x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + x_{\text{S3},\text{C2}} + x_{\text{S4},\text{C2}} \geq 39$
     - $x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + x_{\text{S3},\text{C3}} + x_{\text{S4},\text{C3}} \geq 65$
     - $x_{\text{S1},\text{C4}} + x_{\text{S2},\text{C4}} + x_{\text{S3},\text{C4}} + x_{\text{S4},\text{C4}} \geq 435$

2. **Supply capacity (no plant ships more than its capacity):**
   - $x_{i,\text{C1}} + x_{i,\text{C2}} + x_{i,\text{C3}} + x_{i,\text{C4}} \leq s_i$ for all $i \in I$
   - Explicitly:
     - $x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} + x_{\text{S1},\text{C3}} + x_{\text{S1},\text{C4}} \leq 2531$
     - $x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} + x_{\text{S2},\text{C3}} + x_{\text{S2},\text{C4}} \leq 20$
     - $x_{\text{S3},\text{C1}} + x_{\text{S3},\text{C2}} + x_{\text{S3},\text{C3}} + x_{\text{S3},\text{C4}} \leq 210$
     - $x_{\text{S4},\text{C1}} + x_{\text{S4},\text{C2}} + x_{\text{S4},\text{C3}} + x_{\text{S4},\text{C4}} \leq 241$

3. **Non-negativity:**
   - $x_{ij} \geq 0$ for all $i \in I$, $j \in J$

---

**Complete Model:**

Minimize
\[
\begin{aligned}
&543.756480860856\, x_{\text{S1},\text{C1}} + 23.685276141764653\, x_{\text{S1},\text{C2}} + 23.676386730773032\, x_{\text{S1},\text{C3}} + 447.75143678673766\, x_{\text{S1},\text{C4}} \\
+&883.9151090405642\, x_{\text{S2},\text{C1}} + 0.04977684765576961\, x_{\text{S2},\text{C2}} + 0.0350986687216299\, x_{\text{S2},\text{C3}} + 44.45588531711622\, x_{\text{S2},\text{C4}} \\
+&537.3456896658107\, x_{\text{S3},\text{C1}} + 23.769274659075112\, x_{\text{S3},\text{C2}} + 498.95659249465467\, x_{\text{S3},\text{C3}} + 440.60737890439776\, x_{\text{S3},\text{C4}} \\
+&1791.493192397229\, x_{\text{S4},\text{C1}} + 68.21633865655126\, x_{\text{S4},\text{C2}} + 1432.4837339656747\, x_{\text{S4},\text{C3}} + 1527.7635425462734\, x_{\text{S4},\text{C4}}
\end{aligned}
\]

Subject to:
\[
\begin{aligned}
&x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + x_{\text{S3},\text{C1}} + x_{\text{S4},\text{C1}} \geq 94 \\
&x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + x_{\text{S3},\text{C2}} + x_{\text{S4},\text{C2}} \geq 39 \\
&x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + x_{\text{S3},\text{C3}} + x_{\text{S4},\text{C3}} \geq 65 \\
&x_{\text{S1},\text{C4}} + x_{\text{S2},\text{C4}} + x_{\text{S3},\text{C4}} + x_{\text{S4},\text{C4}} \geq 435 \\
&x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} + x_{\text{S1},\text{C3}} + x_{\text{S1},\text{C4}} \leq 2531 \\
&x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} + x_{\text{S2},\text{C3}} + x_{\text{S2},\text{C4}} \leq 20 \\
&x_{\text{S3},\text{C1}} + x_{\text{S3},\text{C2}} + x_{\text{S3},\text{C3}} + x_{\text{S3},\text{C4}} \leq 210 \\
&x_{\text{S4},\text{C1}} + x_{\text{S4},\text{C2}} + x_{\text{S4},\text{C3}} + x_{\text{S4},\text{C4}} \leq 241 \\
&x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\end{aligned}
\]