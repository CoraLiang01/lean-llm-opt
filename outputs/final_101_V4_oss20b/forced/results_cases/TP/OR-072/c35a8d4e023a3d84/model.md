**Linear‑programming model**

Let  

* \(T=\{1,2,\dots ,24\}\) – the 24 one‑hour shifts of the day.  
* \(d_t\) – the number of staff required in shift \(t\) (taken from the column *Number Required* in *42.csv*).  
* \(x_t\) – the number of drivers/crew members who start working at the beginning of shift \(t\).  
  (Each such person works continuously for 4 consecutive shifts.)

---

### Decision variables
\[
x_t \in \mathbb{Z}_{\ge 0}\qquad \forall\,t\in T
\]

### Objective
\[
\min \;\sum_{t\in T} x_t
\]
(Find the minimum total number of staff that must be scheduled.)

### Constraints
For every shift \(t\in T\) the total number of staff that are on duty during that shift must be at least the required number \(d_t\).  
A staff member who starts at shift \(s\) is on duty in shifts \(s,\,s+1,\,s+2,\,s+3\) (provided those shifts exist).  
Hence

\[
\sum_{\substack{s\in T\\ s\le t\le s+3}} x_s \;\ge\; d_t
\qquad \forall\,t\in T
\]

(If \(s+3>24\) the upper bound of the sum is simply 24; no wrap‑around is assumed.)

---

## Data mapping

| Table ID | Column | Parameter | Mapping rule |
|----------|--------|-----------|--------------|
| `file_0_view_0` | `Shift` | index \(t\) | \(t = \text{Shift}\) (integer 1–24) |
| `file_0_view_0` | `Number Required` | \(d_t\) | \(d_t = \text{Number Required}\) (integer) |

All other columns (`Time`) are informational and not used in the model.  

The model above is a pure integer linear program; if a continuous relaxation is acceptable, replace \(x_t \in \mathbb{Z}_{\ge 0}\) by \(x_t \ge 0\).