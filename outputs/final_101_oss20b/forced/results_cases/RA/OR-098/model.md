**Mathematical model**

Let  

* \(H=\{1,\dots ,150\}\) – set of homeowners (rows of the data).  
* \(W=\{\text{Carpenter},\text{Electrician},\text{Painter},\text{Worker\_004},\dots ,\text{Worker\_150}\}\) – set of workers (columns of the data).  

Parameters (obtained directly from the CSV view `file_0_view_0`)

* \(d_{i,k}\) – number of days worker \(k\) spent renovating homeowner \(i\)’s home.  
  \(\displaystyle d_{i,k}= \text{value in column }k\text{ of row }i\).
* \(\text{owner}_i\) – the worker who owns homeowner \(i\)’s home.  
  \(\displaystyle \text{owner}_i= \text{value in column } \text{Owner}\text{ of row }i\).

Decision variables  

* \(w_k \ge 0\) – daily wage of worker \(k\) (real, unrestricted sign except non‑negativity).

**Objective**  
The problem is a feasibility problem; we may set a trivial objective

\[
\min \; 0 .
\]

**Constraints**

1. **Income–expenditure balance for each homeowner**

\[
\sum_{\substack{k\in W\\ k\neq \text{owner}_i}} d_{i,k}\, w_k
\;=\;
d_{i,\text{owner}_i}\, w_{\text{owner}_i},
\qquad \forall\, i\in H .
\]

2. **Fixed wage for the first worker**

\[
w_{\text{Carpenter}} = 60.00 .
\]

3. **Non‑negativity**

\[
w_k \ge 0 , \qquad \forall\, k\in W .
\]

*(The data already satisfy \(\sum_{i\in H} d_{i,k}=10\) for every \(k\in W\); this is a property of the input and need not be enforced in the model.)*

---

### Data Mapping

| Parameter | Source table | Column name |
|-----------|--------------|-------------|
| \(d_{i,k}\) | `file_0_view_0` | column `k` (e.g., `Carpenter`, `Electrician`, …, `Worker_150`) |
| \(\text{owner}_i\) | `file_0_view_0` | column `Owner` |
| \(w_k\) | decision variable | – |

All indices \(i\) and \(k\) are taken from the complete set of rows and columns returned by the CSVQA call; no sorting or enumeration of values is performed. The model is fully symbolic and ready for a linear‑algebraic solver.