##### Decision Variables

$x_1 \geq 0$: quantity of the first FDK57 car model to fulfill (continuous)  
$x_2 \geq 0$: quantity of the second FDK57 car model to fulfill (continuous)  
$x_3 \geq 0$: quantity of the third FDK57 car model to fulfill (continuous)  

##### Parameters

- Revenue per unit:
  - $r_1 = 119.144$
  - $r_2 = 119.144$
  - $r_3 = 120.144$
- Demand:
  - $d_1 = 30$
  - $d_2 = 40$
  - $d_3 = 50$
- Initial Inventory:
  - $s_1 = 200$
  - $s_2 = 100$
  - $s_3 = 150$

##### Objective Function

$\max\ 119.144\,x_1 + 119.144\,x_2 + 120.144\,x_3$

##### Constraints

1. Demand fulfillment (cannot exceed demand):
   - $x_1 \leq 30$
   - $x_2 \leq 40$
   - $x_3 \leq 50$
2. Inventory limits (cannot exceed initial inventory):
   - $x_1 \leq 200$
   - $x_2 \leq 100$
   - $x_3 \leq 150$
3. Non-negativity:
   - $x_1 \geq 0$
   - $x_2 \geq 0$
   - $x_3 \geq 0$

##### Complete Model

$\begin{align*}
\max\quad & 119.144\,x_1 + 119.144\,x_2 + 120.144\,x_3 \\
\text{s.t.}\quad
& x_1 \leq 30 \\
& x_2 \leq 40 \\
& x_3 \leq 50 \\
& x_1 \leq 200 \\
& x_2 \leq 100 \\
& x_3 \leq 150 \\
& x_1 \geq 0,\ x_2 \geq 0,\ x_3 \geq 0
\end{align*}$

Where:
- $x_1$: quantity fulfilled for the first FDK57 car model (Revenue: 119.144, Demand: 30, Initial Inventory: 200)
- $x_2$: quantity fulfilled for the second FDK57 car model (Revenue: 119.144, Demand: 40, Initial Inventory: 100)
- $x_3$: quantity fulfilled for the third FDK57 car model (Revenue: 120.144, Demand: 50, Initial Inventory: 150)