**Mathematical Model**

Let  

* \(T=\{0,1,\dots ,47\}\) – index of the 48 consecutive half‑hour periods that cover the 24‑hour day.  
* \(S=T\) – index of possible shift start times (each start time is a half‑hour).  

**Parameters**

| Symbol | Meaning | Data source |
|--------|---------|-------------|
| \(R_t\) | Minimum number of waitstaff required in period \(t\) | `file_0_view_0` – column **Requirement** (row index \(t\)) |

**Decision Variables**

| Symbol | Domain | Meaning |
|--------|--------|---------|
| \(y_s\) | \(\mathbb{Z}_{\ge 0}\) | Number of waitstaff whose shift starts at period \(s\) |

**Objective**

\[
\min \;\; \sum_{s\in S} y_s
\]

**Constraints**

For every period \(t\in T\) the total number of staff covering that period must be at least the required level:

\[
\sum_{k=0}^{15} y_{(t-k)\bmod 48} \;\;\ge\;\; R_t
\qquad \forall\, t\in T
\]

*Explanation of the coverage term:*  
A staff member who starts at period \(s\) works continuously for 8 h, i.e., 16 consecutive half‑hour periods.  
Thus period \(t\) is covered by all starts \(s\) such that \(t\) lies in the interval \([s,\,s+15]\) modulo 48.  
The summation over \(k=0\) to \(15\) enumerates exactly those starts.

**Data Mapping**

| Table ID | Column | Parameter |
|----------|--------|-----------|
| `file_0_view_0` | `Requirement` | \(R_t\) (row index \(t\)) |
| `file_0_view_0` | `Time` | (used only for interpretation; not a parameter in the model) |

All other columns in the source file are ignored. The model uses only the `Requirement` values to enforce staffing constraints.