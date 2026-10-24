**Mathematical Model**

Let  

* \(P=\{1,\dots ,80\}\) – set of products (columns \(A1\)–\(A80\)).  
* \(D=\{1,\dots ,22\}\) – set of days in the month.

**Parameters** (all indexed by \(p\in P\))

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(d_p\) | Maximum demand (100 kg units) | `file_0_view_0`, row 1, column \(A_p\) |
| \(p_p\) | Selling price ($/100 kg) | `file_0_view_0`, row 2, column \(A_p\) |
| \(c_p\) | Production cost ($/100 kg) | `file_0_view_0`, row 3, column \(A_p\) |
| \(q_p\) | Daily production quota (max units per day) | `file_0_view_0`, row 4, column \(A_p\) |
| \(a_p\) | Fixed activation cost ($) | `file_1_view_0`, row 1, column \(A_p\) |
| \(b_p\) | Minimum batch size (100 kg units) | `file_2_view_0`, row 1, column \(A_p\) |

**Decision Variables**

| Symbol | Type | Description |
|--------|------|-------------|
| \(x_{p,d}\) | \(\mathbb{Z}_{\ge 0}\) | Units of product \(p\) produced on day \(d\) (100 kg units). |
| \(y_p\) | \(\{0,1\}\) | 1 if product \(p\) is activated (any production in the month), 0 otherwise. |
| \(z_{p,d}\) | \(\{0,1\}\) | 1 if product \(p\) is produced on day \(d\) (enforces batch size). |

**Objective**

\[
\max \;\; \sum_{p\in P}\sum_{d\in D}\bigl(p_p - c_p\bigr)\,x_{p,d}\;-\;\sum_{p\in P} a_p\,y_p
\]

**Constraints**

1. **Demand limits**

\[
\sum_{d\in D} x_{p,d}\;\le\; d_p ,\qquad \forall p\in P
\]

2. **Daily quota limits**

\[
x_{p,d}\;\le\; q_p\,z_{p,d},\qquad \forall p\in P,\; d\in D
\]

3. **Batch‑size enforcement**

\[
x_{p,d}\;\ge\; b_p\,z_{p,d},\qquad \forall p\in P,\; d\in D
\]

4. **Activation linkage**

\[
y_p \;\ge\; z_{p,d},\qquad \forall p\in P,\; d\in D
\]

5. **Variable domains**

\[
x_{p,d}\in \mathbb{Z}_{\ge 0},\; y_p\in\{0,1\},\; z_{p,d}\in\{0,1\},
\qquad \forall p\in P,\; d\in D
\]

---

### Data Mapping

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(d_p\) | `file_0_view_0` | `A1`–`A80` (row 1) |
| \(p_p\) | `file_0_view_0` | `A1`–`A80` (row 2) |
| \(c_p\) | `file_0_view_0` | `A1`–`A80` (row 3) |
| \(q_p\) | `file_0_view_0` | `A1`–`A80` (row 4) |
| \(a_p\) | `file_1_view_0` | `A1`–`A80` (row 1) |
| \(b_p\) | `file_2_view_0` | `A1`–`A80` (row 1) |

All indices, parameters, variables, objective sense, and constraints are defined solely from the original query and the supplied data. No additional data or assumptions are introduced.