**Mathematical model**

Let  

* \(P=\{1,\dots ,80\}\) – set of products (product \(p\) is the column \(A_p\) in the CSV files).  
* \(D=\{1,\dots ,22\}\) – set of production days.  

Parameters (all taken from the CSV files)

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(d_p\) | maximum demand of product \(p\) (100 kg units) | `36‑1.csv`, row “Maximum Demand” |
| \(p_p\) | selling price of product \(p\) ($/100 kg) | `36‑1.csv`, row “Selling Price” |
| \(c_p\) | production cost of product \(p\) ($/100 kg) | `36‑1.csv`, row “Production Cost” |
| \(q_p\) | daily production quota of product \(p\) (max units per day) | `36‑1.csv`, row “Production Quota” |
| \(f_p\) | fixed activation cost of the line for product \(p\) ($) | `36‑2.csv`, row “Activation Cost” |
| \(b_p\) | minimum batch size of product \(p\) (100 kg units) | `36‑3.csv`, row “Minimum Batch Size” |

Decision variables  

* \(x_{p,d}\in \mathbb{Z}_{\ge 0}\) – quantity of product \(p\) produced on day \(d\) (units of 100 kg).  
* \(y_p\in\{0,1\}\) – 1 if the line for product \(p\) is activated, 0 otherwise.  
* \(u_{p,d}\in\{0,1\}\) – 1 if any production of product \(p\) occurs on day \(d\), 0 otherwise.  

**Objective**

\[
\max \;\sum_{p\in P}\sum_{d\in D}\bigl(p_p-c_p\bigr)\,x_{p,d}\;-\;\sum_{p\in P}f_p\,y_p
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{d\in D}x_{p,d}\;\le\;d_p,\qquad \forall p\in P
   \]

2. **Daily quota**  
   \[
   x_{p,d}\;\le\;q_p\,y_p,\qquad \forall p\in P,\;d\in D
   \]

3. **Batch‑size linkage**  
   \[
   x_{p,d}\;\le\;M\,u_{p,d},\qquad
   x_{p,d}\;\ge\;b_p\,u_{p,d},\qquad
   u_{p,d}\;\le\;y_p,\qquad \forall p\in P,\;d\in D
   \]
   where \(M=\max_{p}d_p\) (any sufficiently large upper bound).

4. **Variable domains**  
   \[
   x_{p,d}\in\mathbb{Z}_{\ge0},\; y_p\in\{0,1\},\; u_{p,d}\in\{0,1\}
   \]

---

### Data‑mapping rules

| Parameter | CSV file | Row label | Column | Mapping expression |
|-----------|----------|-----------|--------|---------------------|
| \(d_p\) | `36‑1.csv` | “Maximum Demand (100 kg units)” | `A_p` | `d_p = value(row=0, col=A_p)` |
| \(p_p\) | `36‑1.csv` | “Selling Price ($/100 kg)” | `A_p` | `p_p = value(row=1, col=A_p)` |
| \(c_p\) | `36‑1.csv` | “Production Cost ($/100 kg)” | `A_p` | `c_p = value(row=2, col=A_p)` |
| \(q_p\) | `36‑1.csv` | “Production Quota (max per day)” | `A_p` | `q_p = value(row=3, col=A_p)` |
| \(f_p\) | `36‑2.csv` | “Activation Cost ($)” | `A_p` | `f_p = value(row=0, col=A_p)` |
| \(b_p\) | `36‑3.csv` | “Minimum Batch Size (100 kg units)” | `A_p` | `b_p = value(row=0, col=A_p)` |

All indices \(p\) run over the 80 columns `A1`–`A80`; all days \(d\) run over the 22 production days. No other data are required.