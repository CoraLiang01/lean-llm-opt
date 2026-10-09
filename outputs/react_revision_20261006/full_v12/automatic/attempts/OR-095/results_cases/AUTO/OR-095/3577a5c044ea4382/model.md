## Symbolic Mathematical Model

**Sets**
- $I$: set of widgets, $I = \{\text{Widget1}, \ldots, \text{Widget141}\}$ (from file_1_view_0, column Product)

**Parameters** (from file_1_view_0, columns as indicated)
- $a_i$: labor hours per unit of widget $i$ (LaborHours)
- $b_i$: MaterialA per unit of widget $i$ (MaterialA)
- $c_i$: MaterialB per unit of widget $i$ (MaterialB)
- $p_i$: base profit per unit of widget $i$ (Profit)
- $L$: monthly labor hour limit $= 5000$ (file_0_view_0, Resource=LaborHours, MonthlyLimit)
- $A$: monthly MaterialA limit $= 24000$ (file_0_view_0, Resource=MaterialA, MonthlyLimit)
- $B$: monthly MaterialB limit $= 15000$ (file_0_view_0, Resource=MaterialB, MonthlyLimit)
- $r$: CatalystX byproduct rate for Widget3 $= 5$ kg/unit
- $q$: CatalystX sales price $= 300$ $/kg$
- $q_{disp}$: CatalystX disposal cost $= 200$ $/kg$
- $S$: CatalystX monthly sales cap $= 1500$ kg

**Decision Variables**
- $x_i \geq 0$: number of units of widget $i$ produced, $\forall i \in I$
- $y \geq 0$: kg of CatalystX sold
- $z \geq 0$: kg of CatalystX disposed

**Objective**
\[
\max \left\{
\sum_{i \in I} p_i x_i
+ q y
- q_{disp} z
\right\}
\]

**Constraints**
1. Labor hours:
\[
\sum_{i \in I} a_i x_i \leq L
\]
2. Material A:
\[
\sum_{i \in I} b_i x_i \leq A
\]
3. Material B:
\[
\sum_{i \in I} c_i x_i \leq B
\]
4. CatalystX byproduct balance (only Widget3 produces CatalystX):
\[
r x_{\text{Widget3}} = y + z
\]
5. CatalystX sales cap:
\[
y \leq S
\]
6. Nonnegativity:
\[
x_i \geq 0 \quad \forall i \in I
\]
\[
y \geq 0,\quad z \geq 0
\]

---

## Data Mapping

- $I$: file_1_view_0, column Product
- $a_i$: file_1_view_0, column LaborHours, row $i$
- $b_i$: file_1_view_0, column MaterialA, row $i$
- $c_i$: file_1_view_0, column MaterialB, row $i$
- $p_i$: file_1_view_0, column Profit, row $i$
- $L$: file_0_view_0, Resource=LaborHours, MonthlyLimit
- $A$: file_0_view_0, Resource=MaterialA, MonthlyLimit
- $B$: file_0_view_0, Resource=MaterialB, MonthlyLimit
- $r$: 5 (Widget3 only)
- $q$: 300 (CatalystX sales price)
- $q_{disp}$: 200 (CatalystX disposal cost)
- $S$: 1500 (CatalystX sales cap)
- $x_i$: production quantity of widget $i$
- $y$: CatalystX sold
- $z$: CatalystX disposed

All indices, parameters, and constraints are mapped directly to the provided data.