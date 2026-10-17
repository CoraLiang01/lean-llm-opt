**Mathematical Model**

Let  

* \(W\) – set of all workers (column names of the data file except the column *Owner*).  
* \(w_i\) – daily wage of worker \(i\in W\).  
* \(D_{ij}\) – number of days worker \(j\) spent on the home of worker \(i\) (obtained from the data).  

The model is a feasibility problem; we only need wages that satisfy the balance
conditions.

\[
\begin{aligned}
\text{Minimize}\quad & 0 \\[4pt]
\text{subject to}\quad
& \sum_{\substack{j\in W\\ j\neq i}} D_{ij}\,w_j \;=\; w_i \sum_{j\in W} D_{ji},
&& \forall\, i\in W \quad\text{(income = expenditure)}\\[4pt]
& w_{\text{Carpenter}} \;=\; 60.00 && \text{(fixed wage)}\\[4pt]
& w_i \;\ge\; 0 && \forall\, i\in W .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(D_{ij}\) | Table **file_0_view_0** | For each row \(r\) where `Owner` = \(i\), set \(D_{ij}\) = value in column `j` of that row.  Columns `j` are all column names of the table except `Owner`. |
| \(w_{\text{Carpenter}}\) | Constant | Fixed to 60.00 (given). |

*Indices*  
- \(W = \{\text{Carpenter},\text{Electrician},\text{Painter},\text{Worker\_004},\dots,\text{Worker\_150}\}\).  
- \(i,j \in W\).

All other parameters (e.g., total days per worker) are implicit in the data and need not be modeled explicitly. The model above fully captures the required wage‑balance conditions.