**Mathematical Model**

Let  

* \(W=\{w\}\) be the set of all workers (the column names in *work_days.csv* except the column *Owner*).  
* \(d_{ij}\) be the number of days worker \(j\) spent renovating the home of worker \(i\) (entry in row \(i\), column \(j\) of the file).  
* \(w_j\) be the daily wage of worker \(j\) (decision variable).  

---

### Decision Variables
\[
w_j \;\ge\; 0 \qquad \forall j\in W
\]

### Parameters
\[
d_{ij}\quad\text{(from table `file_0_view_0`, columns `Owner`, `Carpenter`, …, `Worker_150`)}
\]
\[
w_{\text{Carpenter}} = 60.00 \quad\text{(fixed)}
\]

### Constraints

1. **Balance for each homeowner**  
   For every worker \(i\in W\) (who is also a homeowner):
   \[
   \sum_{j\in W} d_{ji}\,w_j \;-\; w_i \sum_{j\in W} d_{ij} \;=\; 0
   \]
   (income from working on others’ homes = expenditure on own home)

2. **Total days worked by each worker**  
   (data consistency – does not involve decision variables)  
   \[
   \sum_{i\in W} d_{ij} \;=\; 10 \qquad \forall j\in W
   \]

3. **Fixed wage for the first worker**  
   \[
   w_{\text{Carpenter}} = 60.00
   \]

### Objective
No objective is required; the model is a feasibility problem.  
(If an objective is desired, one may set \(\min 0\) or \(\max 0\).)

---

## Data Mapping

| Parameter | Source | Table ID | Columns | Description |
|-----------|--------|----------|---------|-------------|
| \(d_{ij}\) | work_days.csv | `file_0_view_0` | `Owner`, `Carpenter`, `Electrician`, `Painter`, `Worker_004`, …, `Worker_150` | Entry at row *i* (owner), column *j* (worker) gives days worked. |
| \(w_{\text{Carpenter}}\) | Fixed | – | – | Daily wage of the first worker (Carpenter) set to 60.00 yuan. |

*All indices \(i,j\) run over the set \(W\) of worker names extracted from the column headers of the file, excluding the column `Owner`.*