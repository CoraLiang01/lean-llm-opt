**Mathematical Model**

Let  

* \(H=\{1,\dots ,150\}\) : set of homeowners (rows of *work_days.csv*)  
* \(W=\{\text{Carpenter},\text{Electrician},\text{Painter},\text{Worker\_004},\dots ,\text{Worker\_150}\}\) : set of workers (columns of *work_days.csv*)  

Parameters (obtained from *work_days.csv*)  

* \(a_{i,w}\) : number of days worker \(w\) spent renovating homeowner \(i\)’s home  
* \(\text{owner}_i\in W\) : worker who owns homeowner \(i\)’s home (value of the *Owner* column for row \(i\))

Decision variables  

* \(w_w\ge 0\) : daily wage of worker \(w\)  (\(w_{\text{Carpenter}}=60.00\) is fixed)

---

### Objective
\[
\min \sum_{w\in W} w_w
\]

(Any feasible wage vector satisfies the problem; minimizing the sum yields a unique solution.)

---

### Constraints  

1. **Total work‑day requirement for each worker**  
   \[
   \sum_{i\in H} a_{i,w}=10 \qquad \forall\, w\in W
   \]

2. **Income–expenditure balance for each homeowner**  
   \[
   w_{\text{owner}_i}\;\sum_{j\in H} a_{j,\text{owner}_i}
   \;=\;
   \sum_{w\in W} a_{i,w}\,w_w
   \qquad \forall\, i\in H
   \]

3. **Fixed wage of the first worker**  
   \[
   w_{\text{Carpenter}} = 60.00
   \]

4. **Non‑negativity**  
   \[
   w_w \ge 0 \qquad \forall\, w\in W
   \]

---

## Data Mapping Rules

| Parameter | Source Table | Column(s) | Notes |
|-----------|---------------|-----------|-------|
| \(a_{i,w}\) | `file_0_view_0` | All columns except `Owner` | Each row \(i\) corresponds to a homeowner; each column \(w\) gives days worked. |
| \(\text{owner}_i\) | `file_0_view_0` | `Owner` | Value is the worker name (e.g., `Carpenter`, `Electrician`, …). |
| \(w_{\text{Carpenter}}\) | Fixed | – | Set to 60.00 yuan. |

All indices and parameters are derived directly from the complete runtime records of *work_days.csv*; no aggregation or omission is performed.