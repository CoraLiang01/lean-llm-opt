##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise.

##### Parameters

- $I = \{\text{S1}, \text{S2}\}$ (Suppliers)
- $J = \{\text{C1}, \text{C2}\}$ (Supermarkets)
- Demands: $d_{\text{C1}} = 144$, $d_{\text{C2}} = 216$
- Fixed costs: $f_{\text{S1}} = 105.97$, $f_{\text{S2}} = 85.31$
- Transportation costs:
  - $c_{\text{S1},\text{C1}} = 2358.39$, $c_{\text{S1},\text{C2}} = 1492.08$
  - $c_{\text{S2},\text{C1}} = 0.07$, $c_{\text{S2},\text{C2}} = 52.32$
- $M = d_{\text{C1}} + d_{\text{C2}} = 360$ (sufficiently large upper bound for each supplier's total shipment)

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

That is,

\[
\min \left[
2358.39\,x_{\text{S1},\text{C1}} + 1492.08\,x_{\text{S1},\text{C2}} + 0.07\,x_{\text{S2},\text{C1}} + 52.32\,x_{\text{S2},\text{C2}}
+ 105.97\,y_{\text{S1}} + 85.31\,y_{\text{S2}}
\right]
\]

##### Constraints

1. **Demand satisfaction (each supermarket's demand must be met):**
   - $x_{\text{S1},j} + x_{\text{S2},j} = d_j$, for $j \in \{\text{C1}, \text{C2}\}$
     - $x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} = 144$
     - $x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} = 216$

2. **Supplier activation (no shipments from inactive suppliers):**
   - $x_{ij} \leq M\,y_i$, for all $i \in I$, $j \in J$
     - $x_{\text{S1},\text{C1}} \leq 360\,y_{\text{S1}}$
     - $x_{\text{S1},\text{C2}} \leq 360\,y_{\text{S1}}$
     - $x_{\text{S2},\text{C1}} \leq 360\,y_{\text{S2}}$
     - $x_{\text{S2},\text{C2}} \leq 360\,y_{\text{S2}}$

3. **Variable domains:**
   - $x_{ij} \geq 0$ (continuous)
   - $y_i \in \{0,1\}$

---

###### Retrieved Information

- **Suppliers:** S1, S2
- **Supermarkets:** C1, C2
- **Demand:** C1: 144, C2: 216
- **Fixed costs:** S1: 105.97, S2: 85.31
- **Transportation costs:**
  - S1 → C1: 2358.39, S1 → C2: 1492.08
  - S2 → C1: 0.07, S2 → C2: 52.32
- **$M = 360$** (sum of all demands)

---

**Complete Mathematical Model:**

\[
\begin{align*}
\min\quad & 2358.39\,x_{\text{S1},\text{C1}} + 1492.08\,x_{\text{S1},\text{C2}} + 0.07\,x_{\text{S2},\text{C1}} + 52.32\,x_{\text{S2},\text{C2}} \\
&\quad + 105.97\,y_{\text{S1}} + 85.31\,y_{\text{S2}} \\
\text{s.t.}\quad
& x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} = 144 \\
& x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} = 216 \\
& x_{\text{S1},\text{C1}} \leq 360\,y_{\text{S1}} \\
& x_{\text{S1},\text{C2}} \leq 360\,y_{\text{S1}} \\
& x_{\text{S2},\text{C1}} \leq 360\,y_{\text{S2}} \\
& x_{\text{S2},\text{C2}} \leq 360\,y_{\text{S2}} \\
& x_{ij} \geq 0,\quad \forall i \in \{\text{S1},\text{S2}\},\ j \in \{\text{C1},\text{C2}\} \\
& y_i \in \{0,1\},\quad \forall i \in \{\text{S1},\text{S2}\}
\end{align*}
\]