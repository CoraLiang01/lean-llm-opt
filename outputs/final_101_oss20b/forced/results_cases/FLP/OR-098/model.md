**Mathematical Model**

Let  

* \(W\) – set of all workers (columns of *work_days.csv* except the first “Owner” column).  
* \(O\) – set of all owners (the values in the “Owner” column).  
  (In the data \(O=W\).)  

Parameters  

* \(D_{ij}\) – number of days worker \(i\in W\) spent renovating owner \(j\in O\).  
  \(D_{ij}\) is read from the cell in row \(j\) and column \(i\) of *work_days.csv*.  
* \(w_{\text{Carpenter}} = 60.00\) – fixed daily wage of the first worker.

Decision variables  

* \(w_i \ge 0\) – daily wage of worker \(i\in W\).

The model is a feasibility problem (no objective).

\[
\begin{aligned}
\text{Find } & \{w_i\}_{i\in W} \\[4pt]
\text{s.t. } & \sum_{\substack{i\in W\\ i\neq j}} w_i\,D_{ij}
               \;=\;
               w_j \sum_{\substack{i\in W\\ i\neq j}} D_{ij},
               && \forall\, j\in O, \\[6pt]
             & w_{\text{Carpenter}} = 60.00, \\[4pt]
             & w_i \ge 0, && \forall\, i\in W .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(D_{ij}\) | *work_days.csv* (table_id: `file_0_view_0`) | For each row with `Owner = j` and each column `i` (excluding the first “Owner” column), set \(D_{ij}\) to the cell value. |
| \(w_{\text{Carpenter}}\) | Fixed value | \(w_{\text{Carpenter}} = 60.00\) |

*Indices*  
* \(W =\) all column names of *work_days.csv* except “Owner”.  
* \(O =\) all distinct values appearing in the “Owner” column (identical to \(W\)).  

No other parameters or constraints are required.