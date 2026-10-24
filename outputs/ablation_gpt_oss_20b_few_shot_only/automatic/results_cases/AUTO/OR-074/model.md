## Mathematical Model

Let  

* \(T=\{1,\dots ,48\}\) – the 48 consecutive half‑hour periods of a 24‑hour day  
  (period 1 = 2:00 am–2:30 am, period 2 = 2:30 am–3:00 am, …, period 48 = 1:30 am–2:00 am).  
* \(R_s\) – the minimum number of waitstaff required in period \(s\) (data from **44.csv**).  
* \(S_t\) – the number of waitstaff whose shift starts at the beginning of period \(t\).  
  Each such staff works continuously for 8 h = 16 consecutive half‑hour periods.

Define the coverage set  

\[
C_t=\{\,s\in T \mid (s-t+48)\bmod 48 \le 15\,\},
\]

i.e. the 16 periods covered by a shift that starts at period \(t\).

### Decision Variables
\[
S_t \;\ge\; 0 \qquad \forall\,t\in T
\]

### Objective
\[
\min \;\sum_{t\in T} S_t
\]

### Constraints
For every period \(s\in T\) the total number of staff covering that period must
meet the required minimum:

\[
\sum_{\substack{t\in T\\ s\in C_t}} S_t \;\ge\; R_s
\qquad \forall\,s\in T
\]

---

## Data Mapping

| File Path | Column | Mapping |
|-----------|--------|---------|
| `Large-scale-or/Others_example/Others2/44.csv` | `Time Requirement` | \(R_s\) for each period \(s\) (row order corresponds to periods 1–48) |

No additional parameters are required. The model uses only the required waitstaff
values from the CSV file and the fixed shift length of 8 h (16 half‑hour periods).