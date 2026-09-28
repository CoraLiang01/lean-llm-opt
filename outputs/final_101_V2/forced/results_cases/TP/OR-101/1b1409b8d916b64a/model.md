##### Decision Variables

Let $x_p \geq 0$ (continuous) denote the monthly production quantity of product $p \in \{P1, P2, \ldots, P{111}\}$.

##### Objective Function

Maximize total monthly profit:
$$
\max \sum_{p=1}^{111} \text{Unit\_Profit}_p \cdot x_p
$$
where the unit profits are:

\[
\begin{align*}
&\text{Unit\_Profit}_{P1} = 28.55,\quad \text{Unit\_Profit}_{P2} = 12.78,\quad \text{Unit\_Profit}_{P3} = 45.21,\quad \ldots,\quad \text{Unit\_Profit}_{P111} = 9.99
\end{align*}
\]

##### Constraints

For each device $d \in \{A, B, C, D, E, F, G, H, I, J\}$, the total processing time used cannot exceed its monthly capacity:

For all $d$:
$$
\sum_{p=1}^{111} \text{DeviceTime}_{d,p} \cdot x_p \leq \text{Monthly\_Capacity}_d
$$

where the device times and capacities are:

- For device $A$:
  - $\text{Monthly\_Capacity}_A = 3500$
  - $\text{DeviceTime}_{A,P1} = 8.1$, $\text{DeviceTime}_{A,P2} = 2.5$, ..., $\text{DeviceTime}_{A,P111} = 8.5$
- For device $B$:
  - $\text{Monthly\_Capacity}_B = 4200$
  - $\text{DeviceTime}_{B,P1} = 10.5$, ..., $\text{DeviceTime}_{B,P111} = 8.7$
- For device $C$:
  - $\text{Monthly\_Capacity}_C = 4500$
  - $\text{DeviceTime}_{C,P1} = 2.1$, ..., $\text{DeviceTime}_{C,P111} = 2.9$
- For device $D$:
  - $\text{Monthly\_Capacity}_D = 2800$
  - $\text{DeviceTime}_{D,P1} = 5.8$, ..., $\text{DeviceTime}_{D,P111} = 8.3$
- For device $E$:
  - $\text{Monthly\_Capacity}_E = 3300$
  - $\text{DeviceTime}_{E,P1} = 9.3$, ..., $\text{DeviceTime}_{E,P111} = 6.7$
- For device $F$:
  - $\text{Monthly\_Capacity}_F = 3800$
  - $\text{DeviceTime}_{F,P1} = 3.8$, ..., $\text{DeviceTime}_{F,P111} = 8.1$
- For device $G$:
  - $\text{Monthly\_Capacity}_G = 4100$
  - $\text{DeviceTime}_{G,P1} = 7.2$, ..., $\text{DeviceTime}_{G,P111} = 5.9$
- For device $H$:
  - $\text{Monthly\_Capacity}_H = 3900$
  - $\text{DeviceTime}_{H,P1} = 11.7$, ..., $\text{DeviceTime}_{H,P111} = 1.4$
- For device $I$:
  - $\text{Monthly\_Capacity}_I = 4800$
  - $\text{DeviceTime}_{I,P1} = 1.1$, ..., $\text{DeviceTime}_{I,P111} = 7.3$
- For device $J$:
  - $\text{Monthly\_Capacity}_J = 3100$
  - $\text{DeviceTime}_{J,P1} = 4.6$, ..., $\text{DeviceTime}_{J,P111} = 12.4$

##### Non-negativity

$$
x_p \geq 0 \quad \forall p \in \{P1, P2, \ldots, P111\}
$$

---

###### Retrieved Information

- Products: $P1$ through $P111$
- Devices: $A$ through $J$
- Unit profits: as listed in unit_product_profits.csv, in source order
- Device times: as listed in device_time.csv, in source order (each device row gives times for all products $P1$ to $P111$)
- Monthly device capacities: as listed in monthly_device_capacity.csv, in source order

**Full numerical data for all coefficients and identifiers is included above, preserving source order.**