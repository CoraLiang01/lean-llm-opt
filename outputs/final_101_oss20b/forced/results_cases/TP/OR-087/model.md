**Mathematical model**

Let  

* \(P=\{1,\dots ,80\}\) – product indices (product \(p\) corresponds to column \(A_p\) in the CSV files).  
* \(D=\{1,\dots ,22\}\) – day indices.

**Parameters**

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(d_p\) | Maximum demand (100 kg units) for product \(p\) | `file_0_view_0`, row 0, column `A_p` |
| \(p_p\) | Selling price ($/100 kg) for product \(p\) | `file_0_view_0`, row 1, column `A_p` |
| \(c_p\) | Production cost ($/100 kg) for product \(p\) | `file_0_view_0`, row 2, column `A_p` |
| \(q_p\) | Production quota (max units per day) for product \(p\) | `file_0_view_0`, row 3, column `A_p` |
| \(b_p\) | Minimum batch size (100 kg units) for product \(p\) | `file_2_view_0`, row 0, column `A_p` |
| \(f_p\) | Activation cost ($) for the production line of product \(p\) | `file_1_view_0`, row 0, column `A_p` |

**Decision variables**

* \(x_{p,d}\in \mathbb{Z}_{+}\) – quantity of product \(p\) produced on day \(d\) (units of 100 kg).  
* \(z_{p,d}\in\{0,1\}\) – 1 if product \(p\) is produced on day \(d\), 0 otherwise.  
* \(y_{p}\in\{0,1\}\) – 1 if the production line for product \(p\) is activated, 0 otherwise.

**Objective**

\[
\max \;\sum_{p\in P}\sum_{d\in D}\bigl(p_p-c_p\bigr)\,x_{p,d}\;-\;\sum_{p\in P} f_p\,y_{p}
\]

**Constraints**

1. **Demand limits**  
   \[
   \sum_{d\in D} x_{p,d}\;\le\; d_p \qquad \forall p\in P
   \]

2. **Daily quota**  
   \[
   x_{p,d}\;\le\; q_p\,z_{p,d} \qquad \forall p\in P,\; d\in D
   \]

3. **Minimum batch size**  
   \[
   x_{p,d}\;\ge\; b_p\,z_{p,d} \qquad \forall p\in P,\; d\in D
   \]

4. **Activation linkage**  
   \[
   y_{p}\;\ge\; z_{p,d} \qquad \forall p\in P,\; d\in D
   \]

5. **Variable domains**  
   \[
   x_{p,d}\in\mathbb{Z}_{+},\; z_{p,d}\in\{0,1\},\; y_{p}\in\{0,1\}
   \]

---

### Data‑mapping rules

| Parameter | CSV file | Row | Column | Notes |
|-----------|----------|-----|--------|-------|
| \(d_p\) | `36-1.csv` | 0 | `A_p` | Maximum demand (100 kg units) |
| \(p_p\) | `36-1.csv` | 1 | `A_p` | Selling price ($/100 kg) |
| \(c_p\) | `36-1.csv` | 2 | `A_p` | Production cost ($/100 kg) |
| \(q_p\) | `36-1.csv` | 3 | `A_p` | Production quota (max per day) |
| \(b_p\) | `36-3.csv` | 0 | `A_p` | Minimum batch size (100 kg units) |
| \(f_p\) | `36-2.csv` | 0 | `A_p` | Activation cost ($) |

All indices \(p\) run over the 80 columns `A1`–`A80`.  
All days \(d\) run over the 22 days of the month.