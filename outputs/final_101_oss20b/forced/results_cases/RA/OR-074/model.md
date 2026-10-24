**Mathematical Model**

Let  

* \(T=\{1,\dots ,48\}\) – the 48 consecutive half‑hour periods that cover the 24‑hour day.  
* \(S=T\) – the set of possible shift start times (a shift can start at any half‑hour).  
* \(R_t\) – the required number of waitstaff in period \(t\) (data from *file_0_view_0*).  
* \(W_{s,t}\in\{0,1\}\) – equals 1 if a shift that starts at period \(s\) is working during period \(t\); otherwise 0.  
  \[
  W_{s,t}= \begin{cases}
  1 & \text{if } 0\le (t-s+48)\bmod 48 \le 15,\\
  0 & \text{otherwise.}
  \end{cases}
  \]
* \(y_s\) – the number of waitstaff who start their 8‑hour shift at period \(s\).

---

### Objective
\[
\min \;\sum_{s\in S} y_s
\]

### Constraints
1. **Coverage of each period**
   \[
   \sum_{s\in S} W_{s,t}\,y_s \;\ge\; R_t , \qquad \forall\, t\in T
   \]

2. **Non‑negativity and integrality**
   \[
   y_s \in \mathbb{Z}_{\ge 0}, \qquad \forall\, s\in S
   \]

---

**Data Mapping**

| Parameter | Source | Column(s) | Mapping |
|-----------|--------|-----------|---------|
| \(R_t\) | `file_0_view_0` | `Requirement` | \(R_t =\) value of `Requirement` for row \(t\) |
| \(W_{s,t}\) | Derived | – | \(W_{s,t}=1\) if \((t-s+48)\bmod 48 \le 15\); otherwise \(0\) |

*No other data are required.*