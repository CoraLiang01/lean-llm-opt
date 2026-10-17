**Symbolic Model**

Let  

* \(P=\{ \text{Widget1},\dots ,\text{Widget141}\}\) – set of widgets.  
* \(x_p\) – number of units of widget \(p\) produced (integer, \(x_p\ge 0\)).  
* \(y\) – kilograms of CatalystX sold (integer, \(y\ge 0\)).  

Parameters (all obtained from the supplied tables)

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(h_p\) | Labor hours per unit of widget \(p\) | `file_1_view_0` – column **LaborHours** |
| \(a_p\) | Material A consumption per unit of widget \(p\) | `file_1_view_0` – column **MaterialA** |
| \(b_p\) | Material B consumption per unit of widget \(p\) | `file_1_view_0` – column **MaterialB** |
| \(\pi_p\) | Base profit per unit of widget \(p\) | `file_1_view_0` – column **Profit** |
| \(L_{\max}\) | Monthly labor‑hour limit | `file_0_view_0` – row where **Resource**=`LaborHours` |
| \(A_{\max}\) | Monthly Material A limit | `file_0_view_0` – row where **Resource**=`MaterialA` |
| \(B_{\max}\) | Monthly Material B limit | `file_0_view_0` – row where **Resource**=`MaterialB` |
| \(C_{\text{sell}}\) | Sale price of CatalystX (per kg) | constant \(300\) |
| \(C_{\text{dispose}}\) | Disposal cost of unsold CatalystX (per kg) | constant \(200\) |
| \(R_{\max}\) | Market cap on CatalystX sales (kg) | constant \(1500\) |
| \(g_3\) | CatalystX generation rate per unit of Widget3 | constant \(5\) |

---

### Objective

\[
\max \;\; \sum_{p\in P} \pi_p\,x_p \;+\; C_{\text{sell}}\;y \;-\; C_{\text{dispose}}\;\bigl(g_3\,x_{\text{Widget3}}-y\bigr)
\]

(Equivalently \( \max \sum_{p}\pi_p x_p + 500\,y - 1000\,x_{\text{Widget3}}\).)

---

### Constraints

1. **Labor limit**  
   \[
   \sum_{p\in P} h_p\,x_p \;\le\; L_{\max}
   \]

2. **Material A limit**  
   \[
   \sum_{p\in P} a_p\,x_p \;\le\; A_{\max}
   \]

3. **Material B limit**  
   \[
   \sum_{p\in P} b_p\,x_p \;\le\; B_{\max}
   \]

4. **CatalystX sales cap**  
   \[
   y \;\le\; R_{\max}
   \]

5. **CatalystX availability**  
   \[
   y \;\le\; g_3\,x_{\text{Widget3}}
   \]

6. **Non‑negativity & integrality**  
   \[
   x_p \in \mathbb{Z}_{\ge 0}\quad \forall p\in P,\qquad
   y \in \mathbb{Z}_{\ge 0}
   \]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(h_p\) | `file_1_view_0` | `LaborHours` |
| \(a_p\) | `file_1_view_0` | `MaterialA` |
| \(b_p\) | `file_1_view_0` | `MaterialB` |
| \(\pi_p\) | `file_1_view_0` | `Profit` |
| \(L_{\max}\) | `file_0_view_0` | `MonthlyLimit` (row **LaborHours**) |
| \(A_{\max}\) | `file_0_view_0` | `MonthlyLimit` (row **MaterialA**) |
| \(B_{\max}\) | `file_0_view_0` | `MonthlyLimit` (row **MaterialB**) |

All other constants (\(C_{\text{sell}}, C_{\text{dispose}}, R_{\max}, g_3\)) are fixed values given in the problem statement.