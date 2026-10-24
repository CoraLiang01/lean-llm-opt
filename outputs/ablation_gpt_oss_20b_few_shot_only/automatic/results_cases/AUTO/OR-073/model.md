## Mathematical Model

**Sets**

- \(P=\{I,\,II,\,III\}\) products  
- \(E=\{A1,\,A2,\,A3,\,B1,\,B2,\,B3,\,B4\}\) equipment

**Parameters**

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(t_{p,e}\) | processing time per unit of product \(p\) on equipment \(e\) | `43.csv` – columns “Product I”, “Product II”, “Product III” |
| \(a_e\) | available operating time of equipment \(e\) | `43.csv` – column “Available Equipment Operating Time” |
| \(c_e\) | equipment cost per unit time on equipment \(e\) | `43.csv` – column “Equipment Cost at Full Load (yuan)” |
| \(p_p\) | selling price per unit of product \(p\) | `43.csv` – column “Unit Price (yuan/unit)” |
| \(r_p\) | raw‑material cost per unit of product \(p\) | `43.csv` – column “Raw Material Cost (yuan/unit)” |
| \(\delta_{p,e}\) | 1 if product \(p\) can be processed on equipment \(e\), 0 otherwise | defined by presence of a processing‑time entry in `43.csv` |

**Decision Variables**

- \(x_{p,e}\ge 0\) units of product \(p\) processed on equipment \(e\)  
  (defined only for \(\delta_{p,e}=1\))

**Objective**

\[
\max \;\sum_{p\in P}\sum_{e\in E}
x_{p,e}\,\bigl(p_p-r_p-c_e\,t_{p,e}\bigr)
\]

**Constraints**

1. **Equipment capacity**

\[
\sum_{p\in P} t_{p,e}\,x_{p,e}\;\le\; a_e
\qquad \forall e\in E
\]

2. **Product‑equipment compatibility**  
   (implicitly enforced by defining \(x_{p,e}\) only when \(\delta_{p,e}=1\))

3. **Non‑negativity**

\[
x_{p,e}\;\ge\;0
\qquad \forall p\in P,\;e\in E
\]

---

## Data Mapping

| Parameter | Source file | Source column | Notes |
|-----------|-------------|---------------|-------|
| \(t_{p,e}\) | `Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv` | “Product I”, “Product II”, “Product III” | If a cell is blank, \(\delta_{p,e}=0\) and the corresponding \(x_{p,e}\) is not defined. |
| \(a_e\) | same file | “Available Equipment Operating Time” | |
| \(c_e\) | same file | “Equipment Cost at Full Load (yuan)” | |
| \(p_p\) | same file | “Unit Price (yuan/unit)” | |
| \(r_p\) | same file | “Raw Material Cost (yuan/unit)” | |
| \(\delta_{p,e}\) | same file | presence of a numeric entry in the time columns | 1 if time entry exists, 0 otherwise. |

All parameters are taken directly from the provided CSV; no additional data are invented. The model uses continuous production quantities and respects the equipment‑specific processing times, capacities, and cost structure described in the problem statement.