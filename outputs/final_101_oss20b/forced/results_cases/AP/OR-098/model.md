**Mathematical model**

Let  

* \(I\) – set of homeowners (rows of *work_days.csv*).  
* \(J\) – set of workers (columns of *work_days.csv* except the column **Owner**).  
* \(d_{ij}\) – number of days worker \(j\) spent renovating homeowner \(i\)’s home.  
* \(o_i\in J\) – the worker who owns homeowner \(i\) (value of the **Owner** column).  
* \(w_j\) – daily wage of worker \(j\) (decision variable).  

Parameters are taken directly from the CSV file; the only fixed parameter is the wage of the first worker listed, **Carpenter**.

\[
\begin{aligned}
\text{Minimize}\quad & 0 \\[4pt]
\text{subject to}\quad
& w_{\text{Carpenter}} = 60.00 \\[4pt]
& \sum_{j\in J\setminus\{o_i\}} d_{ij}\,w_j
   \;=\;
   \sum_{j\in J\setminus\{o_i\}} d_{ij}\,w_{o_i},
   && \forall\, i\in I \quad\text{(income = expenditure)} \\[4pt]
& w_j \;\ge\; 0, && \forall\, j\in J .
\end{aligned}
\]

(If desired, the worker‑day totals \(\sum_{i\in I} d_{ij}=10\) for all \(j\) can be recorded as a data property but need not be enforced in the model.)

---

**Data‑mapping rules**

| Parameter | Source table | Source columns |
|-----------|--------------|----------------|
| \(I\) | `file_0_view_0` | rows (implicit) |
| \(J\) | `file_0_view_0` | columns **Carpenter**, **Electrician**, **Painter**, **Worker_004**, …, **Worker_150** |
| \(d_{ij}\) | `file_0_view_0` | same columns as \(J\); value in row \(i\), column \(j\) |
| \(o_i\) | `file_0_view_0` | column **Owner**; value in row \(i\) |
| \(w_{\text{Carpenter}}\) | fixed | 60.00 |

All other parameters are symbolic and derived from the table; no numerical values are enumerated in the model.