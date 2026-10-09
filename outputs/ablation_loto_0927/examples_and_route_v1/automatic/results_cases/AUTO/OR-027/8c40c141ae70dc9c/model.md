Let $i$ index the Organ products, with the following identifiers and parameters:

- Organic Fruits: Revenue $r_1 = 60.8$, Demand $d_1 = 678{,}906$, Initial Inventory $s_1 = 5{,}034{,}020$
- Organic Staples: Revenue $r_2 = 918.45$, Demand $d_2 = 749{,}927$, Initial Inventory $s_2 = 5{,}589{,}290$
- Organic Vegetables: Revenue $r_3 = 77.52$, Demand $d_3 = 699{,}808$, Initial Inventory $s_3 = 5{,}202{,}710$

Decision variables:
- $x_i$: Number of units of Organ product $i$ to fulfill ($x_i \in \mathbb{Z}_{\geq 0}$)

Objective:
\[
\max \; 60.8\, x_1 + 918.45\, x_2 + 77.52\, x_3
\]

Subject to:
\[
\begin{align*}
& x_1 \leq 678{,}906 \\
& x_1 \leq 5{,}034{,}020 \\
& x_2 \leq 749{,}927 \\
& x_2 \leq 5{,}589{,}290 \\
& x_3 \leq 699{,}808 \\
& x_3 \leq 5{,}202{,}710 \\
& x_1, x_2, x_3 \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Where:
- $x_1$ = units of Organic Fruits fulfilled
- $x_2$ = units of Organic Staples fulfilled
- $x_3$ = units of Organic Vegetables fulfilled

All variables are nonnegative integers. Each $x_i$ cannot exceed both its demand and its initial inventory. The objective is to maximize total revenue from fulfilled Organ product units.