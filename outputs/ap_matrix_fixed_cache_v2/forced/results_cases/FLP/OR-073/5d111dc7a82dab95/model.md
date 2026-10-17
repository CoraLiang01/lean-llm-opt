##### Sets and Indices

- Products: $P = \{\text{I}, \text{II}, \text{III}\}$
- Equipment for procedure A: $A = \{\text{A1}, \text{A2}\}$
- Equipment for procedure B: $B = \{\text{B1}, \text{B2}, \text{B3}\}$

##### Parameters

- Processing time per unit (minutes/unit), $t_{ep}$:
    - For procedure A:
        - $t_{\text{A1},\text{I}} = 5$, $t_{\text{A1},\text{II}} = 10$, $t_{\text{A1},\text{III}}$ undefined (not eligible)
        - $t_{\text{A2},\text{I}} = 7$, $t_{\text{A2},\text{II}} = 9$, $t_{\text{A2},\text{III}} = 12$
    - For procedure B:
        - $t_{\text{B1},\text{I}} = 6$, $t_{\text{B1},\text{II}} = 8$, $t_{\text{B1},\text{III}}$ undefined
        - $t_{\text{B2},\text{I}} = 4$, $t_{\text{B2},\text{II}}$ undefined, $t_{\text{B2},\text{III}} = 11$
        - $t_{\text{B3},\text{I}} = 7$, $t_{\text{B3},\text{II}}$ undefined, $t_{\text{B3},\text{III}}$ undefined

- Available equipment operating time (minutes), $T_e$:
    - $T_{\text{A1}} = 6000$
    - $T_{\text{A2}} = 10000$
    - $T_{\text{B1}} = 4000$
    - $T_{\text{B2}} = 7000$
    - $T_{\text{B3}} = 4000$

- Equipment cost at full load (yuan), $C_e$:
    - $C_{\text{A1}} = 300$
    - $C_{\text{A2}} = 321$
    - $C_{\text{B1}} = 250$
    - $C_{\text{B2}} = 783$
    - $C_{\text{B3}} = 200$

- Raw material cost per unit, $r_p$:
    - $r_{\text{I}} = 0.25$
    - $r_{\text{II}} = 0.35$
    - $r_{\text{III}} = 0.5$

- Selling price per unit, $s_p$:
    - $s_{\text{I}} = 1.25$
    - $s_{\text{II}} = 2$
    - $s_{\text{III}} = 2.8$

##### Decision Variables

- $x_p \geq 0$: quantity of product $p$ produced, $p \in P$ (continuous)
- $x_{ep}^A \geq 0$: quantity of product $p$ processed on equipment $e$ for procedure A
- $x_{ep}^B \geq 0$: quantity of product $p$ processed on equipment $e$ for procedure B

##### Eligibility Constraints

- For procedure A:
    - Product I: $e \in \{\text{A1}, \text{A2}\}$
    - Product II: $e \in \{\text{A1}, \text{A2}\}$
    - Product III: $e = \text{A2}$
- For procedure B:
    - Product I: $e \in \{\text{B1}, \text{B2}, \text{B3}\}$
    - Product II: $e = \text{B1}$
    - Product III: $e = \text{B2}$

##### Objective Function

\[
\max \left\{
\sum_{p \in P} \left[ s_p x_p - r_p x_p \right]
- \sum_{e \in A \cup B} C_e \cdot \frac{1}{T_e} \sum_{p} t_{ep} x_{ep}
\right\}
\]

where $x_{ep}$ is $x_{ep}^A$ for $e \in A$, $x_{ep}^B$ for $e \in B$, and the sum over $p$ is only for eligible products on equipment $e$.

##### Constraints

1. **Production assignment for each product and procedure:**
    - For procedure A:
        - $x_p = \sum_{e \in A_p} x_{ep}^A$, for all $p \in P$
    - For procedure B:
        - $x_p = \sum_{e \in B_p} x_{ep}^B$, for all $p \in P$
    - Where $A_p$ and $B_p$ are the eligible equipment sets for product $p$ in procedures A and B, respectively.

2. **Equipment time constraints:**
    - For each $e \in A$:
        - $\sum_{p \in P_e^A} t_{ep} x_{ep}^A \leq T_e$
    - For each $e \in B$:
        - $\sum_{p \in P_e^B} t_{ep} x_{ep}^B \leq T_e$
    - Where $P_e^A$ and $P_e^B$ are the sets of products eligible for equipment $e$ in procedures A and B, respectively.

3. **Nonnegativity:**
    - $x_p \geq 0$, $x_{ep}^A \geq 0$, $x_{ep}^B \geq 0$ for all eligible $(e,p)$.

##### Explicit Data

- **Procedure A:**
    - $A = \{\text{A1}, \text{A2}\}$
    - $t_{\text{A1},\text{I}} = 5$, $t_{\text{A1},\text{II}} = 10$
    - $t_{\text{A2},\text{I}} = 7$, $t_{\text{A2},\text{II}} = 9$, $t_{\text{A2},\text{III}} = 12$
    - $T_{\text{A1}} = 6000$, $C_{\text{A1}} = 300$
    - $T_{\text{A2}} = 10000$, $C_{\text{A2}} = 321$

- **Procedure B:**
    - $B = \{\text{B1}, \text{B2}, \text{B3}\}$
    - $t_{\text{B1},\text{I}} = 6$, $t_{\text{B1},\text{II}} = 8$
    - $t_{\text{B2},\text{I}} = 4$, $t_{\text{B2},\text{III}} = 11$
    - $t_{\text{B3},\text{I}} = 7$
    - $T_{\text{B1}} = 4000$, $C_{\text{B1}} = 250$
    - $T_{\text{B2}} = 7000$, $C_{\text{B2}} = 783$
    - $T_{\text{B3}} = 4000$, $C_{\text{B3}} = 200$

- **Raw material cost:** $r_{\text{I}} = 0.25$, $r_{\text{II}} = 0.35$, $r_{\text{III}} = 0.5$
- **Selling price:** $s_{\text{I}} = 1.25$, $s_{\text{II}} = 2$, $s_{\text{III}} = 2.8$

##### Model Summary

Let $x_{\text{I}}, x_{\text{II}}, x_{\text{III}}$ be the production quantities.

Let $x_{\text{A1,I}}^A, x_{\text{A1,II}}^A, x_{\text{A2,I}}^A, x_{\text{A2,II}}^A, x_{\text{A2,III}}^A$ be the assignments for procedure A.

Let $x_{\text{B1,I}}^B, x_{\text{B1,II}}^B, x_{\text{B2,I}}^B, x_{\text{B2,III}}^B, x_{\text{B3,I}}^B$ be the assignments for procedure B.

\[
\begin{align*}
\max\ & \left[1.25 x_{\text{I}} + 2 x_{\text{II}} + 2.8 x_{\text{III}} - 0.25 x_{\text{I}} - 0.35 x_{\text{II}} - 0.5 x_{\text{III}}\right] \\
& - 300 \frac{1}{6000}(5 x_{\text{A1,I}}^A + 10 x_{\text{A1,II}}^A)
- 321 \frac{1}{10000}(7 x_{\text{A2,I}}^A + 9 x_{\text{A2,II}}^A + 12 x_{\text{A2,III}}^A) \\
& - 250 \frac{1}{4000}(6 x_{\text{B1,I}}^B + 8 x_{\text{B1,II}}^B)
- 783 \frac{1}{7000}(4 x_{\text{B2,I}}^B + 11 x_{\text{B2,III}}^B)
- 200 \frac{1}{4000}(7 x_{\text{B3,I}}^B)
\end{align*}
\]

Subject to:

\[
\begin{align*}
& x_{\text{I}} = x_{\text{A1,I}}^A + x_{\text{A2,I}}^A = x_{\text{B1,I}}^B + x_{\text{B2,I}}^B + x_{\text{B3,I}}^B \\
& x_{\text{II}} = x_{\text{A1,II}}^A + x_{\text{A2,II}}^A = x_{\text{B1,II}}^B \\
& x_{\text{III}} = x_{\text{A2,III}}^A = x_{\text{B2,III}}^B \\
& 5 x_{\text{A1,I}}^A + 10 x_{\text{A1,II}}^A \leq 6000 \\
& 7 x_{\text{A2,I}}^A + 9 x_{\text{A2,II}}^A + 12 x_{\text{A2,III}}^A \leq 10000 \\
& 6 x_{\text{B1,I}}^B + 8 x_{\text{B1,II}}^B \leq 4000 \\
& 4 x_{\text{B2,I}}^B + 11 x_{\text{B2,III}}^B \leq 7000 \\
& 7 x_{\text{B3,I}}^B \leq 4000 \\
& x_{\text{A1,I}}^A, x_{\text{A1,II}}^A, x_{\text{A2,I}}^A, x_{\text{A2,II}}^A, x_{\text{A2,III}}^A \geq 0 \\
& x_{\text{B1,I}}^B, x_{\text{B1,II}}^B, x_{\text{B2,I}}^B, x_{\text{B2,III}}^B, x_{\text{B3,I}}^B \geq 0 \\
& x_{\text{I}}, x_{\text{II}}, x_{\text{III}} \geq 0
\end{align*}
\]

All variables are continuous and nonnegative. Only eligible assignments are included per the product-equipment eligibility rules.