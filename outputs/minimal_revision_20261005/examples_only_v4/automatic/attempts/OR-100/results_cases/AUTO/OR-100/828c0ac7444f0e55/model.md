**Abstract Mathematical Model**

**Index Sets**
- $I$: set of components, indexed by $i$ (from all Unnamed: 0 in file_1_view_0 and columns C1–C111 in file_0_view_0)
- $K$: set of workshops, indexed by $k$ (from all Unnamed: 0 in file_0_view_0 and workshop in file_2_view_0)

**Parameters**
- $p_i$: unit price of component $i$  
  Data: file_1_view_0, columns Unnamed: 0 (component ID), unit_price
- $a_{ki}$: unit processing time of component $i$ in workshop $k$  
  Data: file_0_view_0, rows Unnamed: 0 (workshop), columns C1–C111 (component ID)
- $b_k$: total available working hours in workshop $k$  
  Data: file_2_view_0, columns workshop, total_hours

**Decision Variables**
- $x_i$: number of units to produce of component $i$  
  Domain: $x_i \in \mathbb{Z}_{\geq 0}$ for all $i \in I$

**Objective**
\[
\max \sum_{i \in I} p_i x_i
\]

**Constraints**
\[
\sum_{i \in I} a_{ki} x_i \leq b_k \qquad \forall k \in K
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
\]

---

**Data Mapping**

- $I$: All component IDs from file_1_view_0.Unnamed: 0 and file_0_view_0 columns C1–C111
- $K$: All workshop names from file_0_view_0.Unnamed: 0 and file_2_view_0.workshop
- $p_i$: file_1_view_0, columns Unnamed: 0 (component ID), unit_price
- $a_{ki}$: file_0_view_0, rows Unnamed: 0 (workshop), columns C1–C111 (component ID)
- $b_k$: file_2_view_0, columns workshop, total_hours

---

**Summary:**  
Choose integer production quantities $x_i$ for each component $i$ to maximize total output value, subject to workshop time limits. Each workshop's total processing time for all components produced cannot exceed its available hours. All data and index sets are mapped directly from the supplied files and columns.