**Mathematical model**

Let  

* \(W=\{w\}\) be the set of all workers (the column names in *work_days.csv* except the column **Owner**).  
* For each pair \((i,j)\in W\times W\) let  

\[
d_{ij}\in\mathbb{N}
\]

be the number of days worker \(j\) spent renovating homeowner \(i\)’s house.  
(The value is taken from the cell in row *Owner = i* and column *j* of the CSV.)

* For each worker \(j\in W\) let  

\[
t_j = \sum_{i\in W} d_{ij}
\]

be the total number of days contributed by worker \(j\) (given to be 10 for every \(j\)).

Decision variables  

\[
w_j \in \mathbb{R}_{\ge 0}\qquad \forall j\in W
\]

( daily wage of worker \(j\) ).

---

### Constraints

1. **Fixed wage of the first worker**  

\[
w_{\text{Carpenter}} = 60.00
\]

2. **Balance of income and expenditure for every homeowner**  

For each homeowner \(i\in W\),

\[
\sum_{\substack{j\in W\\ j\neq i}} d_{ij}\,(w_j - w_i)=0
\]

( income from working on others’ homes equals expenditure on own home).

3. **Total days contributed by each worker**  

\[
\sum_{i\in W} d_{ij}=10 \qquad \forall j\in W
\]

( data consistency; can be omitted if the data already satisfy it).

4. **Non‑negativity**  

\[
w_j \ge 0 \qquad \forall j\in W
\]

---

### Objective

No optimisation objective is required; the model is a feasibility problem.  
If an objective is needed, one may minimise a constant, e.g. \(\min 0\).

---

## Data Mapping (CSVQA_DATA)

| Parameter | Source table | Source column | Mapping rule |
|-----------|--------------|---------------|--------------|
| \(d_{ij}\) | `file_0_view_0` | column *j* (any worker column) | For each row where `Owner = i`, set \(d_{ij}\) to the cell value in column *j*. |
| \(t_j\) | `file_0_view_0` | column *j* | \(t_j = \sum_{i} d_{ij}\) (should equal 10 for all \(j\)). |
| Fixed wage | – | – | \(w_{\text{Carpenter}} = 60.00\) (hard‑coded). |

All other data (e.g., the set \(W\)) are derived directly from the column names of the CSV, excluding the **Owner** column. No additional data are introduced.