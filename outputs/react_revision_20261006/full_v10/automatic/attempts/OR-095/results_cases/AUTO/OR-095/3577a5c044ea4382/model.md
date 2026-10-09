Mathematical Model

Sets:
- $I$: set of widgets, $I = \{\text{Widget1}, \ldots, \text{Widget141}\}$ (from file_1_view_0, column Product)
- $R$: set of resources, $R = \{\text{LaborHours}, \text{MaterialA}, \text{MaterialB}\}$ (from file_0_view_0, column Resource)

Parameters (Data Mapping):
- $a_{i,r}$: amount of resource $r \in R$ required per unit of widget $i \in I$ (from file_1_view_0, columns LaborHours, MaterialA, MaterialB)
- $p_i$: base profit per unit of widget $i$ (from file_1_view_0, column Profit)
- $L_r$: monthly limit for resource $r$ (from file_0_view_0, column MonthlyLimit)
- $c$: CatalystX generated per unit of Widget3, $c = 5$ kg/unit (problem statement)
- $v$: sale price of CatalystX, $v = 300$ $/kg$ (problem statement)
- $d$: disposal cost of CatalystX, $d = 200$ $/kg$ (problem statement)
- $S$: maximum CatalystX sales per month, $S = 1500$ kg (problem statement)

Decision Variables:
- $x_i \geq 0$: number of units of widget $i \in I$ to produce (continuous)
- $y \geq 0$: amount of CatalystX (kg) sold (continuous)
- $z \geq 0$: amount of CatalystX (kg) disposed (continuous)

Objective:
Maximize total profit:
\[
\max \sum_{i \in I} p_i x_i + v y - d z
\]

Constraints:
1. Resource limits (for each $r \in R$):
\[
\sum_{i \in I} a_{i,r} x_i \leq L_r
\]
where $a_{i,\text{LaborHours}}$ = LaborHours, $a_{i,\text{MaterialA}}$ = MaterialA, $a_{i,\text{MaterialB}}$ = MaterialB from file_1_view_0.

2. CatalystX balance:
\[
c \cdot x_{\text{Widget3}} = y + z
\]

3. CatalystX sales cap:
\[
y \leq S
\]

4. Nonnegativity:
\[
x_i \geq 0 \quad \forall i \in I
\]
\[
y \geq 0
\]
\[
z \geq 0
\]

Data Mapping

- $I$ (widgets): file_1_view_0, column Product
- $R$ (resources): file_0_view_0, column Resource
- $a_{i,r}$: file_1_view_0, columns LaborHours, MaterialA, MaterialB (for each $i$ and $r$)
- $p_i$: file_1_view_0, column Profit
- $L_r$: file_0_view_0, column MonthlyLimit
- $c$: 5 (problem statement, Widget3 only)
- $v$: 300 (problem statement)
- $d$: 200 (problem statement)
- $S$: 1500 (problem statement)
- $x_i$: decision variable for each $i \in I$
- $y$: decision variable, CatalystX sold
- $z$: decision variable, CatalystX disposed

Special notes:
- Only Widget3 generates CatalystX; $x_{\text{Widget3}}$ is the production quantity of Widget3.
- All other widgets do not generate CatalystX.
- All variables are continuous and nonnegative.

End of model.