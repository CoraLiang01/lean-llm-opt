**Mathematical model**

Let  

* \(W=\{\,\text{Carpenter},\text{Electrician},\text{Painter},\text{Worker\_004},\dots ,\text{Worker\_150}\,\}\) – set of workers (columns of the CSV).  
* \(H=\{1,\dots ,150\}\) – set of homeowners (rows of the CSV).  
* \(d_{i,h}\) – number of days worker \(i\in W\) spent renovating homeowner \(h\in H\)’s home (parameter).  
* \(t_i=\sum_{h\in H} d_{i,h}\) – total days contributed by worker \(i\) (parameter, equal to 10 for every \(i\)).  
* \(o_h\in W\) – the worker type that owns homeowner \(h\) (value in the *Owner* column of row \(h\)).  
* \(w_i\ge 0\) – daily wage of worker \(i\) (decision variable).  

The model is a feasibility problem; we set a dummy objective.

\[
\begin{aligned}
\min\;& 0 \\[4pt]
\text{s.t. } & \sum_{i\in W} d_{i,h}\,w_i \;=\; w_{o_h}\,\bigl(t_{o_h}-d_{o_h,h}\bigr) ,
&& \forall\,h\in H \quad\text{(balance for homeowner \(h\))} \\[4pt]
& w_{\text{Carpenter}} \;=\; 60.00 \quad\text{(fixed wage)} \\[4pt]
& w_i \;\ge\; 0 , && \forall\,i\in W .
\end{aligned}
\]

---

**Data‑mapping rules**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(d_{i,h}\) | `work_days.csv` (table_id `file_0_view_0`) | For each row \(h\) (row index 0‑based) and each column \(i\in W\) (excluding *Owner*), read the numeric value in that cell. |
| \(t_i\) | `work_days.csv` | For each worker \(i\in W\), sum the values of column \(i\) over all rows \(h\). |
| \(o_h\) | `work_days.csv` | For each row \(h\), read the value in the *Owner* column; this string is the worker type belonging to \(h\). |
| Fixed wage | Query | \(w_{\text{Carpenter}} = 60.00\) yuan. |

All other data (e.g., total days per worker) are derived from the same table and are not required as separate inputs. The model uses only the indices and parameters defined above.