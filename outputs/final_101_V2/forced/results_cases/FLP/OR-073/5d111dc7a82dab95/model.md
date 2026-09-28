##### Sets

- Products: $P = \{\text{I}, \text{II}, \text{III}\}$
- Procedure A equipment: $A = \{\text{A1}, \text{A2}\}$
- Procedure B equipment: $B = \{\text{B1}, \text{B2}, \text{B3}\}$

##### Parameters

- Processing time (hours/unit) for product $p$ on equipment $e$: $t_{e,p}$
- Available equipment operating time (hours) for equipment $e$: $T_e$
- Equipment cost at full load (yuan) for equipment $e$: $C_e$
- Raw material cost per unit for product $p$: $r_p$
- Selling price per unit for product $p$: $s_p$

From the data:

| Equipment | Product I | Product II | Product III | Available Time | Equipment Cost at Full Load |
|-----------|-----------|------------|-------------|---------------|----------------------------|
| A1        | 5         | 10         | —           | 6000          | 300                        |
| A2        | 7         | 9          | 12          | 10000         | 321                        |
| B1        | 6         | 8          | —           | 4000          | 250                        |
| B2        | 4         | —          | 11          | 7000          | 783                        |
| B3        | 7         | —          | —           | 4000          | 200                        |

Raw material cost: $r_{\text{I}} = 0.25$, $r_{\text{II}} = 0.35$, $r_{\text{III}} = 0.5$

Selling price: $s_{\text{I}} = 1.25$, $s_{\text{II}} = 2$, $s_{\text{III}} = 2.8$

##### Decision Variables

- $x_p \geq 0$: production quantity of product $p$ (continuous)
- $y_{e,p} \geq 0$: quantity of product $p$ processed on equipment $e$ (continuous)

##### Objective Function

Maximize total profit:
\[
\max \left[
\sum_{p \in P} s_p x_p
- \sum_{p \in P} r_p x_p
- \sum_{e} C_e \cdot \frac{1}{T_e} \sum_{p} t_{e,p} y_{e,p}
\right]
\]

##### Constraints

1. **Production flow for each product and procedure:**

   - For each product, the amount processed in procedure A must equal the amount processed in procedure B, and both equal the total produced:
     - Product I:
       \[
       x_{\text{I}} = y_{\text{A1},\text{I}} + y_{\text{A2},\text{I}}
       \]
       \[
       x_{\text{I}} = y_{\text{B1},\text{I}} + y_{\text{B2},\text{I}} + y_{\text{B3},\text{I}}
       \]
     - Product II:
       \[
       x_{\text{II}} = y_{\text{A1},\text{II}} + y_{\text{A2},\text{II}}
       \]
       \[
       x_{\text{II}} = y_{\text{B1},\text{II}}
       \]
     - Product III:
       \[
       x_{\text{III}} = y_{\text{A2},\text{III}}
       \]
       \[
       x_{\text{III}} = y_{\text{B2},\text{III}}
       \]

2. **Equipment time capacity:**

   For each equipment $e$:
   \[
   \sum_{p} t_{e,p} y_{e,p} \leq T_e
   \]
   where $t_{e,p}$ is defined only if product $p$ can be processed on equipment $e$ (otherwise, $y_{e,p} = 0$).

   Specifically, from the data:

   - A1: $t_{\text{A1},\text{I}} = 5$, $t_{\text{A1},\text{II}} = 10$, $t_{\text{A1},\text{III}}$ not defined ($y_{\text{A1},\text{III}} = 0$), $T_{\text{A1}} = 6000$
   - A2: $t_{\text{A2},\text{I}} = 7$, $t_{\text{A2},\text{II}} = 9$, $t_{\text{A2},\text{III}} = 12$, $T_{\text{A2}} = 10000$
   - B1: $t_{\text{B1},\text{I}} = 6$, $t_{\text{B1},\text{II}} = 8$, $t_{\text{B1},\text{III}}$ not defined ($y_{\text{B1},\text{III}} = 0$), $T_{\text{B1}} = 4000$
   - B2: $t_{\text{B2},\text{I}} = 4$, $t_{\text{B2},\text{II}}$ not defined ($y_{\text{B2},\text{II}} = 0$), $t_{\text{B2},\text{III}} = 11$, $T_{\text{B2}} = 7000$
   - B3: $t_{\text{B3},\text{I}} = 7$, $t_{\text{B3},\text{II}}$ not defined ($y_{\text{B3},\text{II}} = 0$), $t_{\text{B3},\text{III}}$ not defined ($y_{\text{B3},\text{III}} = 0$), $T_{\text{B3}} = 4000$

   So, for each equipment:
   - A1: $5y_{\text{A1},\text{I}} + 10y_{\text{A1},\text{II}} \leq 6000$
   - A2: $7y_{\text{A2},\text{I}} + 9y_{\text{A2},\text{II}} + 12y_{\text{A2},\text{III}} \leq 10000$
   - B1: $6y_{\text{B1},\text{I}} + 8y_{\text{B1},\text{II}} \leq 4000$
   - B2: $4y_{\text{B2},\text{I}} + 11y_{\text{B2},\text{III}} \leq 7000$
   - B3: $7y_{\text{B3},\text{I}} \leq 4000$

3. **Processing restrictions (enforced by variable domains):**

   - $y_{\text{A1},\text{III}} = 0$, $y_{\text{B1},\text{III}} = 0$, $y_{\text{B2},\text{II}} = 0$, $y_{\text{B3},\text{II}} = 0$, $y_{\text{B3},\text{III}} = 0$, $y_{\text{A2},\text{I}} \geq 0$, etc., as per the data and description.

##### Full Mathematical Model

Let the variables be:

- $x_{\text{I}}, x_{\text{II}}, x_{\text{III}} \geq 0$
- $y_{\text{A1},\text{I}}, y_{\text{A1},\text{II}} \geq 0$
- $y_{\text{A2},\text{I}}, y_{\text{A2},\text{II}}, y_{\text{A2},\text{III}} \geq 0$
- $y_{\text{B1},\text{I}}, y_{\text{B1},\text{II}} \geq 0$
- $y_{\text{B2},\text{I}}, y_{\text{B2},\text{III}} \geq 0$
- $y_{\text{B3},\text{I}} \geq 0$

All other $y_{e,p}$ are zero.

**Objective:**
\[
\max \Bigg[
1.25x_{\text{I}} + 2x_{\text{II}} + 2.8x_{\text{III}}
- 0.25x_{\text{I}} - 0.35x_{\text{II}} - 0.5x_{\text{III}}
- \left(
\frac{300}{6000}(5y_{\text{A1},\text{I}} + 10y_{\text{A1},\text{II}})
+ \frac{321}{10000}(7y_{\text{A2},\text{I}} + 9y_{\text{A2},\text{II}} + 12y_{\text{A2},\text{III}})
+ \frac{250}{4000}(6y_{\text{B1},\text{I}} + 8y_{\text{B1},\text{II}})
+ \frac{783}{7000}(4y_{\text{B2},\text{I}} + 11y_{\text{B2},\text{III}})
+ \frac{200}{4000}(7y_{\text{B3},\text{I}})
\right)
\Bigg]
\]

**Subject to:**

- $x_{\text{I}} = y_{\text{A1},\text{I}} + y_{\text{A2},\text{I}}$
- $x_{\text{I}} = y_{\text{B1},\text{I}} + y_{\text{B2},\text{I}} + y_{\text{B3},\text{I}}$
- $x_{\text{II}} = y_{\text{A1},\text{II}} + y_{\text{A2},\text{II}}$
- $x_{\text{II}} = y_{\text{B1},\text{II}}$
- $x_{\text{III}} = y_{\text{A2},\text{III}}$
- $x_{\text{III}} = y_{\text{B2},\text{III}}$

- $5y_{\text{A1},\text{I}} + 10y_{\text{A1},\text{II}} \leq 6000$
- $7y_{\text{A2},\text{I}} + 9y_{\text{A2},\text{II}} + 12y_{\text{A2},\text{III}} \leq 10000$
- $6y_{\text{B1},\text{I}} + 8y_{\text{B1},\text{II}} \leq 4000$
- $4y_{\text{B2},\text{I}} + 11y_{\text{B2},\text{III}} \leq 7000$
- $7y_{\text{B3},\text{I}} \leq 4000$

- $y_{\text{A1},\text{III}} = 0$, $y_{\text{B1},\text{III}} = 0$, $y_{\text{B2},\text{II}} = 0$, $y_{\text{B3},\text{II}} = 0$, $y_{\text{B3},\text{III}} = 0$

- All variables $\geq 0$ and continuous.

##### Parameters (from CSV)

- $t_{\text{A1},\text{I}} = 5$, $t_{\text{A1},\text{II}} = 10$, $T_{\text{A1}} = 6000$, $C_{\text{A1}} = 300$
- $t_{\text{A2},\text{I}} = 7$, $t_{\text{A2},\text{II}} = 9$, $t_{\text{A2},\text{III}} = 12$, $T_{\text{A2}} = 10000$, $C_{\text{A2}} = 321$
- $t_{\text{B1},\text{I}} = 6$, $t_{\text{B1},\text{II}} = 8$, $T_{\text{B1}} = 4000$, $C_{\text{B1}} = 250$
- $t_{\text{B2},\text{I}} = 4$, $t_{\text{B2},\text{III}} = 11$, $T_{\text{B2}} = 7000$, $C_{\text{B2}} = 783$
- $t_{\text{B3},\text{I}} = 7$, $T_{\text{B3}} = 4000$, $C_{\text{B3}} = 200$
- $r_{\text{I}} = 0.25$, $r_{\text{II}} = 0.35$, $r_{\text{III}} = 0.5$
- $s_{\text{I}} = 1.25$, $s_{\text{II}} = 2$, $s_{\text{III}} = 2.8$

All coefficients and constraints are directly from the CSV data and the problem description.