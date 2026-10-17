**Mathematical model**

Let  

* \(P\) – set of all products (index \(p\)).  
* \(x_p\) – continuous production quantity of product \(p\) (units).  

Parameters (obtained from *41.csv*)  

| Symbol | Meaning | Source column |
|--------|---------|---------------|
| \(l_p\) | labor required per unit of product \(p\) | “Labor per unit” |
| \(m_p\) | material required per unit of product \(p\) | “Material per unit” |
| \(s_p\) | selling price per unit of product \(p\) | “Selling Price” |
| \(v_p\) | variable cost per unit of product \(p\) | “Variable Cost” |

Fixed weekly operating cost  

\[
F = 4500
\]

Weekly capacity limits  

\[
L = 1650 \quad\text{(labor units)} ,\qquad
M = 1850 \quad\text{(material units)}
\]

---

### Decision variables
\[
x_p \;\ge\; 0 , \qquad \forall p\in P
\]

### Objective (maximise weekly net profit)
\[
\max \;\; \sum_{p\in P} (s_p - v_p)\,x_p \;-\; F
\]

### Constraints
1. **Labor capacity**  
   \[
   \sum_{p\in P} l_p\,x_p \;\le\; L
   \]
2. **Material capacity**  
   \[
   \sum_{p\in P} m_p\,x_p \;\le\; M
   \]

---

**Data‑mapping rules**

| Table ID | Column name in CSV | Parameter name |
|----------|--------------------|----------------|
| `file_0_view_0` | “Product Name” | index \(p\) |
| `file_0_view_0` | “Labor per unit” | \(l_p\) |
| `file_0_view_0` | “Material per unit” | \(m_p\) |
| `file_0_view_0` | “Selling Price” | \(s_p\) |
| `file_0_view_0` | “Variable Cost” | \(v_p\) |

*Each row of the table defines one product \(p\) and its associated parameter values.*  

The model uses all 198 products returned by the observation; no additional aggregation or filtering is applied.