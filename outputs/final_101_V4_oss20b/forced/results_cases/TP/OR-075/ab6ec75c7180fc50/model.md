**Mathematical Model**

Let  

* \(I=\{1,2,\dots ,110\}\) be the set of projects (index = *Project ID*).  
* \(c_i\) – capital requirement of project \(i\) (k$).  
* \(n_i\) – expected NPV of project \(i\) (k$).  
* \(y_i\in\{0,1\}\) – decision variable: 1 if project \(i\) is selected, 0 otherwise.

\[
\begin{aligned}
\max\;&\sum_{i\in I} n_i\,y_i \\
\text{s.t. } &\sum_{i\in I} c_i\,y_i \;\le\; 1000 \qquad\text{(budget)}\\[4pt]
& y_4 + y_7 \;\le\; 1 \qquad\text{(mutually exclusive)}\\[4pt]
& y_6 \;\le\; y_1 \qquad\text{(pre‑requisite)}\\[4pt]
& y_{10} \;\le\; y_5 \qquad\text{(contingent)}\\[4pt]
& y_i \in \{0,1\}\quad \forall i\in I .
\end{aligned}
\]

---

**Data Mapping Rules**

| Index Set | Source | Identifier | Mapping |
|-----------|--------|------------|---------|
| \(I\) | `file_0_view_0` | `Project ID` | `I = {1,…,110}` |
| Parameter `c_i` | `file_0_view_0` | `Capital (k$)` | `c[i] = value of "Capital (k$)" for project i` |
| Parameter `n_i` | `file_0_view_0` | `NPV (k$)` | `n[i] = value of "NPV (k$)" for project i` |
| Decision variable `y_i` | – | – | binary variable for each project i |

All numerical values for `c_i` and `n_i` are taken directly from the rows of `project.csv` (table `file_0_view_0`). No additional data or aggregation is required.