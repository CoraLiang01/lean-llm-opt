Let $x_i$ be the total quantity (in 100 kg units) of product $i$ produced in the month, for $i \in \{A1, A2, ..., A80\}$.

Let $y_i$ be a binary variable indicating whether the production line for product $i$ is activated ($y_i = 1$) or not ($y_i = 0$).

Parameters (from the data):

- $d_i$: Maximum demand for product $i$ (100 kg units)
- $p_i$: Selling price for product $i$ ($/100 kg)
- $c_i$: Production cost for product $i$ ($/100 kg)
- $q_i$: Production quota for product $i$ (max per day, 100 kg units)
- $f_i$: Fixed activation cost for product $i$ ($)
- $m_i$: Minimum batch size for product $i$ (100 kg units)
- $T = 22$: Number of production days in the month

All data is indexed by product $i \in \{A1, ..., A80\}$, with values as given in the retrieved CSVs.

---

**Objective:**

\[
\max \sum_{i} \left[ (p_i - c_i) x_i - f_i y_i \right]
\]

---

**Constraints:**

1. **Demand constraint:**
   \[
   x_i \leq d_i \qquad \forall i
   \]

2. **Monthly production quota constraint:**
   \[
   x_i \leq q_i \cdot T \qquad \forall i
   \]

3. **Minimum batch size (if produced):**
   \[
   x_i \geq m_i y_i \qquad \forall i
   \]

4. **Non-negativity and integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i
   \]
   \[
   y_i \in \{0,1\} \qquad \forall i
   \]

---

**Parameter values (for all $i$):**

| $i$  | $d_i$ | $p_i$ | $c_i$ | $q_i$ | $f_i$ | $m_i$ |
|------|-------|-------|-------|-------|-------|-------|
| A1   | 5732  | 111   | 71.38 | 586   | 178539| 18    |
| A2   | 5607  | 81    | 45.02 | 329   | 157708| 25    |
| A3   | 4653  | 145   | 107.04| 541   | 85192 | 23    |
| ...  | ...   | ...   | ...   | ...   | ...   | ...   |
| A80  | 4466  | 106   | 67.19 | 556   | 146842| 18    |

(Use all 80 products and their corresponding parameter values as retrieved.)

---

**Full Model:**

\[
\begin{align*}
\max \quad & \sum_{i = A1}^{A80} \left[ (p_i - c_i) x_i - f_i y_i \right] \\
\text{s.t.} \quad
& x_i \leq d_i \qquad \forall i = A1, ..., A80 \\
& x_i \leq q_i \cdot 22 \qquad \forall i = A1, ..., A80 \\
& x_i \geq m_i y_i \qquad \forall i = A1, ..., A80 \\
& x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i = A1, ..., A80 \\
& y_i \in \{0,1\} \qquad \forall i = A1, ..., A80 \\
\end{align*}
\]

Where all parameter values are as given in the retrieved data, for each product $i$.