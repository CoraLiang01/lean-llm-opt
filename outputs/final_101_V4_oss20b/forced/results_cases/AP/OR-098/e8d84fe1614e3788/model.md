**Mathematical Model**

Let  

* \(W\) – set of all workers (the values that appear in the *Owner* column).  
* \(d_{ij}\) – number of days worker \(j\) spent renovating the home of worker \(i\).  
* \(w_i\) – daily wage of worker \(i\) (decision variable).  

The data are taken from the CSV file **work_days.csv** (view id `file_0_view_0`).

---

### Variables
\[
w_i \;\ge\; 0 \qquad \forall i\in W
\]

### Parameters
\[
d_{ij} \;\ge\; 0 \qquad \forall i,j\in W
\]

### Constraints
1. **Fixed wage for the first worker (Carpenter)**  
   \[
   w_{\text{Carpenter}} = 60.00
   \]

2. **Balance of income and expenditure for each worker**  
   For every worker \(i\in W\):
   \[
   \Bigl(\sum_{j\in W} d_{ji}\Bigr)\, w_i \;=\; \sum_{j\in W} d_{ij}\, w_j
   \]
   (income earned on others’ homes equals wages paid for work on own home).

3. **Non‑negativity**  
   \[
   w_i \;\ge\; 0 \qquad \forall i\in W
   \]

No objective function is required; the system of equations determines the wages.

---

**Data Mapping**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(d_{ij}\) | `file_0_view_0` | For each worker \(i\in W\) (row where `Owner = i`), the value in column `j` gives \(d_{ij}\). |
| \(w_{\text{Carpenter}}\) | Fixed | Set to 60.00 as given. |

*Index set \(W\)* is the set of all distinct values that appear in the `Owner` column of `file_0_view_0` (e.g., `Carpenter`, `Electrician`, `Painter`, `Worker_004`, …, `Worker_150`).  
*All other columns of the CSV are treated as the worker identifiers \(j\).*