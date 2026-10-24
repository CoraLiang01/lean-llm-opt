**Symbolic Model**

Let  

* \(W\) – set of workers (columns of *work_days.csv* except the first column *Owner*).  
* \(H\) – set of homeowners (rows of *work_days.csv*; the same set as \(W\)).  

Parameters (derived from the data)

\[
D_{h,i}\;=\;\text{days worker }i\text{ spent on homeowner }h
\qquad (h\in H,\; i\in W)
\]

\[
a_i \;=\; \sum_{\substack{h\in H\\ h\neq i}} D_{h,i}
\qquad (i\in W)
\]

\[
b_{i,j}\;=\;D_{i,j}\qquad (i,j\in W,\; i\neq j)
\]

Decision variables  

\[
w_i \;\ge\; 0 \qquad (i\in W)
\]

*Fixed wage*  

\[
w_{\text{Carpenter}} \;=\; 60.00
\]

*Balance constraints* – for every worker \(i\)

\[
a_i\,w_i \;=\; \sum_{\substack{j\in W\\ j\neq i}} b_{i,j}\,w_j
\qquad (i\in W)
\]

*Objective* – feasibility only (e.g. minimize \(0\)).  

\[
\min \; 0
\]

---

**Data Mapping**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(D_{h,i}\) | `file_0_view_0` | Cell in row where **Owner** = \(h\), column = worker \(i\) |
| \(a_i\) | `file_0_view_0` | Sum of all cells in column \(i\) except the row where **Owner** = \(i\) |
| \(b_{i,j}\) | `file_0_view_0` | Same as \(D_{i,j}\) for \(i\neq j\) |
| \(w_{\text{Carpenter}}\) | Fixed | 60.00 |

*Indices*  

* \(W =\) {Carpenter, Electrician, Painter, Worker_004, … , Worker_150}  
* \(H = W\)

All parameters are read directly from the CSV; no numerical values are listed here.