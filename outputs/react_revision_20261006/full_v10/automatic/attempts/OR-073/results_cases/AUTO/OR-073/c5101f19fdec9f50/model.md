Mathematical Model

Sets
P = {I, II, III}         // Products
A = {A1, A2}             // Equipment for procedure A
B = {B1, B2, B3}         // Equipment for procedure B

Parameters (from file_0_view_0)
t_{a,p} = processing time per unit of product p on equipment a ∈ A (from "Product p" column, row a)
t_{b,p} = processing time per unit of product p on equipment b ∈ B (from "Product p" column, row b)
T_a = available operating time for equipment a ∈ A ("Available Equipment Operating Time", row a)
T_b = available operating time for equipment b ∈ B ("Available Equipment Operating Time", row b)
C_a = equipment cost at full load for equipment a ∈ A ("Equipment Cost at Full Load (yuan)", row a)
C_b = equipment cost at full load for equipment b ∈ B ("Equipment Cost at Full Load (yuan)", row b)
rmc_p = raw material cost per unit of product p ("Raw Material Cost (yuan/unit)", row 7)
sp_p = selling price per unit of product p ("Unit Price (yuan/unit)", row 8)

Feasibility sets (from question and data)
A_p = allowed A equipment for product p:
  A_I = {A1, A2}
  A_II = {A1, A2}
  A_III = {A2}
B_p = allowed B equipment for product p:
  B_I = {B1, B2, B3}
  B_II = {B1}
  B_III = {B2}

Decision Variables
x_p ≥ 0 : total production quantity of product p ∈ P
y_{a,p} ≥ 0 : quantity of product p processed on equipment a ∈ A_p (procedure A)
z_{b,p} ≥ 0 : quantity of product p processed on equipment b ∈ B_p (procedure B)

Objective
Maximize total profit:
max ∑_{p∈P} [sp_p x_p - rmc_p x_p] - ∑_{a∈A} C_a (∑_{p∈P: a∈A_p} t_{a,p} y_{a,p} / T_a) - ∑_{b∈B} C_b (∑_{p∈P: b∈B_p} t_{b,p} z_{b,p} / T_b)

Constraints

1. Production assignment (procedure A):
For all p ∈ P:
 x_p = ∑_{a∈A_p} y_{a,p}

2. Production assignment (procedure B):
For all p ∈ P:
 x_p = ∑_{b∈B_p} z_{b,p}

3. Equipment A time limits:
For all a ∈ A:
 ∑_{p∈P: a∈A_p} t_{a,p} y_{a,p} ≤ T_a

4. Equipment B time limits:
For all b ∈ B:
 ∑_{p∈P: b∈B_p} t_{b,p} z_{b,p} ≤ T_b

5. Non-negativity:
x_p ≥ 0 ∀ p ∈ P
y_{a,p} ≥ 0 ∀ a ∈ A_p, p ∈ P
z_{b,p} ≥ 0 ∀ b ∈ B_p, p ∈ P

Data Mapping

- file_0_view_0, rows 0-1: A1, A2 (A equipment), with "Product I", "Product II", "Product III" columns for t_{a,p}, "Available Equipment Operating Time" for T_a, "Equipment Cost at Full Load (yuan)" for C_a
- file_0_view_0, rows 3-5: B1, B2, B3 (B equipment), with "Product I", "Product II", "Product III" columns for t_{b,p}, "Available Equipment Operating Time" for T_b, "Equipment Cost at Full Load (yuan)" for C_b
- file_0_view_0, row 7: "Raw Material Cost (yuan/unit)" for rmc_p
- file_0_view_0, row 8: "Unit Price (yuan/unit)" for sp_p

Feasibility sets for each product and equipment are as described above, based on the question and data.

All variables are continuous and nonnegative. The model maximizes profit, accounting for raw material costs, equipment costs (proportional to usage), and equipment and process assignment constraints.