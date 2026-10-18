Let $x_i$ be the number of 100 kg units of product $i$ produced in the month (integer, $x_i \geq 0$), and $y_i$ be a binary variable indicating whether production line $i$ is activated ($y_i \in \{0,1\}$). All indices and coefficients are as in the retrieved data, with $i$ ranging over $A1$ to $A80$.

**Parameters (for each product $i$):**

- $D_i$: Maximum Demand (100 kg units)
- $P_i$: Selling Price ($/100 kg$)
- $C_i$: Production Cost ($/100 kg$)
- $Q_i$: Production Quota (max per day, 100 kg units)
- $F_i$: Activation Cost ($)
- $B_i$: Minimum Batch Size (100 kg units)
- $T$: Number of production days in the month ($T = 22$)

**Data (source order, original row order):**

- Maximum Demand (100 kg units):  
  $D_{A1}=5732$, $D_{A2}=5607$, ..., $D_{A80}=4466$
- Selling Price ($/100 kg$):  
  $P_{A1}=111$, $P_{A2}=81$, ..., $P_{A80}=106$
- Production Cost ($/100 kg$):  
  $C_{A1}=71.38$, $C_{A2}=45.02$, ..., $C_{A80}=67.19$
- Production Quota (max per day):  
  $Q_{A1}=586$, $Q_{A2}=329$, ..., $Q_{A80}=556$
- Activation Cost ($):  
  $F_{A1}=178539$, $F_{A2}=157708$, ..., $F_{A80}=146842$
- Minimum Batch Size (100 kg units):  
  $B_{A1}=18$, $B_{A2}=25$, ..., $B_{A80}=18$

---

**Mathematical Model**

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$, for $i = A1, ..., A80$ (number of 100 kg units of product $i$ produced)
- $y_i \in \{0,1\}$, for $i = A1, ..., A80$ (1 if line $i$ is activated, 0 otherwise)

**Objective:**
\[
\max \sum_{i=A1}^{A80} \left[ (P_i - C_i) x_i - F_i y_i \right]
\]

**Subject to:**

1. **Demand constraint (do not exceed demand):**
   \[
   x_i \leq D_i, \quad \forall i = A1, ..., A80
   \]

2. **Monthly production quota constraint (do not exceed total possible production):**
   \[
   x_i \leq Q_i \cdot T, \quad \forall i = A1, ..., A80
   \]

3. **Minimum batch size and activation logic:**
   \[
   x_i \geq B_i y_i, \quad \forall i = A1, ..., A80
   \]
   \[
   x_i \leq (Q_i \cdot T) y_i, \quad \forall i = A1, ..., A80
   \]
   (If $y_i = 0$, then $x_i = 0$; if $y_i = 1$, then $x_i \geq B_i$)

4. **Variable domains:**
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad y_i \in \{0,1\}, \quad \forall i = A1, ..., A80
   \]

---

**Explicit Data (source order):**

- For $i = A1$ to $A80$:

| Product | $D_i$ | $P_i$ | $C_i$ | $Q_i$ | $F_i$ | $B_i$ |
|---------|-------|-------|-------|-------|-------|-------|
| A1 | 5732 | 111 | 71.38 | 586 | 178539 | 18 |
| A2 | 5607 | 81 | 45.02 | 329 | 157708 | 25 |
| A3 | 4653 | 145 | 107.04 | 541 | 85192 | 23 |
| ... | ... | ... | ... | ... | ... | ... |
| A80 | 4466 | 106 | 67.19 | 556 | 146842 | 18 |

---

**Summary of Model:**

\[
\begin{align*}
\max \quad & \sum_{i=A1}^{A80} \left[ (P_i - C_i) x_i - F_i y_i \right] \\
\text{s.t.} \quad & x_i \leq D_i, \quad \forall i \\
& x_i \leq Q_i \cdot 22, \quad \forall i \\
& x_i \geq B_i y_i, \quad \forall i \\
& x_i \leq (Q_i \cdot 22) y_i, \quad \forall i \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad y_i \in \{0,1\}, \quad \forall i
\end{align*}
\]

All coefficients and identifiers are as retrieved above, in original file and row order.