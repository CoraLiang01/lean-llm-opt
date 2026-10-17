## Mathematical Model

**Sets and parameters**

Let \(T=\{1,\ldots,24\}\) be the hourly periods, with period 1 starting at 00:00.
Let \(r_t\) be the required number of staff in period \(t\). Each staff member works four consecutive hours; coverage wraps around midnight.

**Decision variables**

\[
x_t\in\mathbb{Z}_{\geq0}\qquad(t\in T),
\]

where \(x_t\) is the number of staff starting work at the beginning of period \(t\).

**Objective**

\[
\min\ \sum_{t\in T}x_t.
\]

**Coverage constraints**

\[
\sum_{k=0}^{3}x_{1+((t-k-1)\bmod24)}\geq r_t
\qquad(t\in T).
\]

**Data Mapping**

| Parameter | Table | Column |
|---|---|---|
| Period order | `file_0_view_0` | `Time`, in source-row order |
| \(r_t\) | `file_0_view_0` | `Number Required` |

The 24 records are the 24 consecutive hourly periods. Period indices are one-based; solution variable names `C0` through `C23` follow that same creation order.
