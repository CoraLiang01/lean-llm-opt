**Mathematical Model**

Let  

* \(P=\{I,II,III\}\) – product types  
* \(E=\{A1,A2,B1,B2,B3\}\) – equipment types  

Define the admissible equipment for each product  

\[
E_I=\{A1,A2,B1,B2,B3\},\qquad
E_{II}=\{A1,A2,B1\},\qquad
E_{III}=\{A2,B2\}.
\]

---

### Parameters  

| Symbol | Description | Source |
|--------|-------------|--------|
| \(\text{Time}_{p,e}\) | Processing time per unit of product \(p\) on equipment \(e\) | `file_0_view_0`, column `"Product I"` if \(p=I\); `"Product II"` if \(p=II\); `"Product III"` if \(p=III\) |
| \(\text{Avail}_e\) | Available operating time of equipment \(e\) | `file_0_view_0`, column `"Available Equipment Operating Time"` |
| \(\text{Cost}_e\) | Equipment cost at full load (yuan) for equipment \(e\) | `file_0_view_0`, column `"Equipment Cost at Full Load (yuan)"` |
| \(\text{Profit}_p\) | Profit per unit of product \(p\) (selling price – raw‑material cost) | *Not provided in data* (query‑defined parameter) |

---

### Decision Variables  

\[
x_{p,e}\;\;(\ge 0,\;\text{continuous})\qquad
\forall\,p\in P,\; e\in E_p
\]

\(x_{p,e}\) is the number of units of product \(p\) produced on equipment \(e\).

---

### Objective  

\[
\max \;\sum_{p\in P}\sum_{e\in E_p}
\Bigl(\text{Profit}_p - \text{Cost}_e\,\text{Time}_{p,e}\Bigr)\,x_{p,e}
\]

---

### Constraints  

1. **Equipment‑time limits**  
   \[
   \sum_{p\in P_e}\text{Time}_{p,e}\,x_{p,e}\;\le\;\text{Avail}_e
   \qquad\forall\,e\in E
   \]

2. **Non‑negativity**  
   \[
   x_{p,e}\;\ge\;0
   \qquad\forall\,p\in P,\; e\in E_p
   \]

---

**Data Mapping Rules**

| Parameter | Table ID | Column | Row Key |
|-----------|----------|--------|---------|
| \(\text{Time}_{p,e}\) | `file_0_view_0` | `"Product I"` / `"Product II"` / `"Product III"` | `"Equipment / Cost" = e` |
| \(\text{Avail}_e\) | `file_0_view_0` | `"Available Equipment Operating Time"` | `"Equipment / Cost" = e` |
| \(\text{Cost}_e\) | `file_0_view_0` | `"Equipment Cost at Full Load (yuan)"` | `"Equipment / Cost" = e` |
| \(\text{Profit}_p\) | *Query‑defined* | – | – |

All other data (e.g., raw‑material cost, selling price) are not present in the supplied tables and are treated as external parameters.