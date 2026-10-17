**Mathematical model**

Let  

* \(P=\{1,\dots ,80\}\) – product indices (column A1 … A80)  
* \(D=\{1,\dots ,22\}\) – production days  

Decision variables  

\[
x_{p,d}\in \mathbb{Z}_{\ge 0}\qquad\text{(units of 100 kg of product }p\text{ on day }d)
\]
\[
y_{p}\in\{0,1\}\qquad\text{(1 if the line for product }p\text{ is activated)}
\]

Parameters (all taken directly from the CSV files)

* \(demand_p\) – maximum demand (units of 100 kg) – from **file_0_view_0**, row 0, column Ap  
* \(price_p\) – selling price ($/100 kg) – from **file_0_view_0**, row 1, column Ap  
* \(cost_p\) – production cost ($/100 kg) – from **file_0_view_0**, row 2, column Ap  
* \(quota_p\) – daily production quota if the line is dedicated – from **file_0_view_0**, row 3, column Ap  
* \(actcost_p\) – fixed activation cost – from **file_1_view_0**, row 0, column Ap  
* \(minbatch_p\) – minimum batch size (units of 100 kg) – from **file_2_view_0**, row 0, column Ap  

The model is

\[
\begin{aligned}
\max\;&\sum_{p\in P}\sum_{d\in D}\bigl(price_p - cost_p\bigr)\,x_{p,d}
      \;-\;\sum_{p\in P} actcost_p\,y_{p} \\[4pt]
\text{s.t. } &\sum_{d\in D} x_{p,d}\;\ge\; demand_p , &&\forall p\in P \quad\text{(demand)}\\[2pt]
&x_{p,d}\;\le\; quota_p\,y_{p} , &&\forall p\in P,\;d\in D \quad\text{(quota)}\\[2pt]
&x_{p,d}\;\ge\; minbatch_p\,y_{p} , &&\forall p\in P,\;d\in D \quad\text{(minimum batch)}\\[2pt]
&x_{p,d}\in\mathbb{Z}_{\ge0}, &&\forall p\in P,\;d\in D\\
&y_{p}\in\{0,1\}, &&\forall p\in P
\end{aligned}
\]

---

### Data‑mapping rules

| Parameter | Source table | Row | Column | Description |
|-----------|--------------|-----|--------|-------------|
| \(demand_p\) | `file_0_view_0` | 0 | `Ap` | Max demand (units of 100 kg) |
| \(price_p\) | `file_0_view_0` | 1 | `Ap` | Selling price ($/100 kg) |
| \(cost_p\) | `file_0_view_0` | 2 | `Ap` | Production cost ($/100 kg) |
| \(quota_p\) | `file_0_view_0` | 3 | `Ap` | Daily quota if line dedicated |
| \(actcost_p\) | `file_1_view_0` | 0 | `Ap` | Fixed activation cost |
| \(minbatch_p\) | `file_2_view_0` | 0 | `Ap` | Minimum batch size (units of 100 kg) |

*All product indices \(p\) correspond to column `Ap` (e.g., `A1` → p = 1, `A80` → p = 80).*

The model above is ready for numerical solution with the supplied data.