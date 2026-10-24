**Model**

Let  

* \(E=\{A1,A2,A3,B1,B2\}\) – equipment types  
* \(P=\{I,II,III\}\) – products  

Decision variables  
\[
x_{e,i}\;\;(\text{units of product }i\text{ processed on equipment }e),\qquad
x_{e,i}\ge 0
\]

Parameters  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(t_{e,i}\) | processing time per unit of product \(i\) on equipment \(e\) | `43.csv` – column “Product I/II/III” |
| \(T_e\) | available operating time of equipment \(e\) | `43.csv` – column “Available Equipment Operating Time” |
| \(c_e\) | equipment cost per hour of equipment \(e\) | `43.csv` – column “Equipment Cost at Full Load (yuan)” |
| \(a_{e,i}\) | 1 if product \(i\) can be processed on equipment \(e\), 0 otherwise | defined below |
| \(\pi_i\) | profit per unit of product \(i\) (selling price – raw‑material cost) | user‑defined |

**Allowed processing matrix \(a_{e,i}\)**  

\[
a_{e,i}=\begin{cases}
1 & \text{if }e\in\{A1,A2,A3,B1,B2\}\text{ and }i=I,\\[4pt]
1 & \text{if }e\in\{A1,A2,A3\}\text{ and }i=II,\\[4pt]
1 & \text{if }e=A2\text{ and }i=III,\\[4pt]
1 & \text{if }e=B2\text{ and }i=III,\\[4pt]
0 & \text{otherwise.}
\end{cases}
\]

(Thus \(a_{B1,II}=1\), \(a_{B1,III}=0\), \(a_{A3,III}=0\), etc.)

**Objective**

\[
\max \;\; \sum_{i\in P}\pi_i\!\!\sum_{e\in E}x_{e,i}
\;-\;
\sum_{e\in E}c_e\!\!\sum_{i\in P}t_{e,i}\,x_{e,i}
\]

**Constraints**

1. **Equipment‑time limits**  
   \[
   \sum_{i\in P} t_{e,i}\,x_{e,i}\;\le\;T_e,
   \qquad \forall e\in E
   \]

2. **Processing‑feasibility**  
   \[
   x_{e,i}\;\le\;M\,a_{e,i},\qquad \forall e\in E,\;i\in P
   \]
   (with \(M\) a sufficiently large constant; equivalently, define \(x_{e,i}\) only for pairs with \(a_{e,i}=1\).)

3. **Non‑negativity**  
   \[
   x_{e,i}\;\ge\;0,\qquad \forall e\in E,\;i\in P
   \]

---

### Data Mapping

| Parameter | Table ID | Column | Notes |
|-----------|----------|--------|-------|
| \(t_{e,i}\) | `file_0_view_0` (and its duplicates) | “Product I”, “Product II”, “Product III” | Value is processing time per unit on equipment \(e\). |
| \(T_e\) | `file_0_view_0` | “Available Equipment Operating Time” | Total operating time available for equipment \(e\). |
| \(c_e\) | `file_0_view_0` | “Equipment Cost at Full Load (yuan)” | Cost per hour of operating equipment \(e\). |
| \(a_{e,i}\) | – | – | Defined by the product‑equipment compatibility rules in the problem statement. |
| \(\pi_i\) | – | – | User‑specified profit per unit for each product. |

*All other data (e.g., raw‑material cost, selling price) are encapsulated in \(\pi_i\) and are not directly read from the CSV.*