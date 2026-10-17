**Mathematical Model**

Let  

* \(I\) – set of all generation lot options (rows of *energy.csv*).  
* \(x_i \in \mathbb{Z}_{\ge 0}\) – number of lots of option \(i\) to purchase.  

Parameters (obtained from *energy.csv*)  

* \(g_i\) – generation per lot for option \(i\)  \((\text{gen\_per\_lot})\).  
* \(c_i\) – cost per lot for option \(i\)  \((\text{cost\_per\_lot})\).  
* \(\text{tech}_i\) – technology type of option \(i\)  \((\text{tech})\).  

Demand requirement  

* Total demand \(D = 200\).

\[
\begin{aligned}
\min\;& \sum_{i\in I} c_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} g_i\,x_i \;\ge\; D \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Table ID | Column in CSV | Parameter / Index |
|----------|---------------|-------------------|
| `file_0_view_0` | `option` | index \(i \in I\) |
| `file_0_view_0` | `tech` | parameter \(\text{tech}_i\) |
| `file_0_view_0` | `gen_per_lot` | parameter \(g_i\) |
| `file_0_view_0` | `cost_per_lot` | parameter \(c_i\) |

All rows returned in the observation constitute the full set \(I\); no additional filtering is applied beyond the initial tech filter specified in the query. No other constraints or capacities are introduced.