Let:
- \( x_{ij} \): Number of units of product \( j \) placed on display (shelf) \( i \).
- \( i \in \{1,2,\ldots,10\} \) (ShelfID from capacity.csv)
- \( j \in \{1,2,\ldots,20\} \) (ProductName from products.csv, in original order)

Define:
- \( v_j \): Value of product \( j \) (from products.csv)
- \( w_j \): Weight of product \( j \) (from products.csv)
- \( C_i \): Capacity of shelf \( i \) (from capacity.csv)

**Product Index Mapping (from products.csv, in order):**
1. Smartphone
2. Laptop
3. Headphones
4. Camera
5. Smartwatch
6. Tablet
7. Bluetooth Speaker
8. Keyboard
9. Mouse
10. Monitor
11. Printer
12. External Hard Drive
13. Router
14. Power Bank
15. Memory Card
16. USB Flash Drive
17. Smart Home Hub
18. Gaming Console
19. Fitness Tracker
20. E-Reader

**Shelf Index Mapping (from capacity.csv, in order):**
1. Shelf 1 (Capacity 5.0)
2. Shelf 2 (Capacity 7.0)
3. Shelf 3 (Capacity 6.0)
4. Shelf 4 (Capacity 8.0)
5. Shelf 5 (Capacity 5.5)
6. Shelf 6 (Capacity 9.0)
7. Shelf 7 (Capacity 6.5)
8. Shelf 8 (Capacity 7.5)
9. Shelf 9 (Capacity 8.2)
10. Shelf 10 (Capacity 5.7)

---

### Mathematical Model

**Decision Variables:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\]

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
\]
where \( v_j \) is as follows (in order):
\[
\begin{align*}
v_1 &= 200 &\text{(Smartphone)} \\
v_2 &= 1500 &\text{(Laptop)} \\
v_3 &= 100 &\text{(Headphones)} \\
v_4 &= 800 &\text{(Camera)} \\
v_5 &= 250 &\text{(Smartwatch)} \\
v_6 &= 600 &\text{(Tablet)} \\
v_7 &= 150 &\text{(Bluetooth Speaker)} \\
v_8 &= 80 &\text{(Keyboard)} \\
v_9 &= 50 &\text{(Mouse)} \\
v_{10} &= 300 &\text{(Monitor)} \\
v_{11} &= 400 &\text{(Printer)} \\
v_{12} &= 120 &\text{(External Hard Drive)} \\
v_{13} &= 60 &\text{(Router)} \\
v_{14} &= 40 &\text{(Power Bank)} \\
v_{15} &= 30 &\text{(Memory Card)} \\
v_{16} &= 25 &\text{(USB Flash Drive)} \\
v_{17} &= 100 &\text{(Smart Home Hub)} \\
v_{18} &= 500 &\text{(Gaming Console)} \\
v_{19} &= 90 &\text{(Fitness Tracker)} \\
v_{20} &= 180 &\text{(E-Reader)} \\
\end{align*}
\]

**Constraints:**

1. **Shelf Capacity Constraints** (for each shelf \( i \)):
   \[
   \sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
   \]
   where \( w_j \) is as follows (in order):
\[
\begin{align*}
w_1 &= 1.0 &\text{(Smartphone)} \\
w_2 &= 5.0 &\text{(Laptop)} \\
w_3 &= 0.5 &\text{(Headphones)} \\
w_4 &= 2.0 &\text{(Camera)} \\
w_5 &= 0.3 &\text{(Smartwatch)} \\
w_6 &= 1.5 &\text{(Tablet)} \\
w_7 &= 1.0 &\text{(Bluetooth Speaker)} \\
w_8 &= 0.8 &\text{(Keyboard)} \\
w_9 &= 0.2 &\text{(Mouse)} \\
w_{10} &= 3.0 &\text{(Monitor)} \\
w_{11} &= 4.0 &\text{(Printer)} \\
w_{12} &= 0.5 &\text{(External Hard Drive)} \\
w_{13} &= 0.3 &\text{(Router)} \\
w_{14} &= 0.4 &\text{(Power Bank)} \\
w_{15} &= 0.05 &\text{(Memory Card)} \\
w_{16} &= 0.02 &\text{(USB Flash Drive)} \\
w_{17} &= 0.6 &\text{(Smart Home Hub)} \\
w_{18} &= 4.0 &\text{(Gaming Console)} \\
w_{19} &= 0.2 &\text{(Fitness Tracker)} \\
w_{20} &= 0.5 &\text{(E-Reader)} \\
\end{align*}
\]
   and \( C_i \) is:
\[
\begin{align*}
C_1 &= 5.0 \\
C_2 &= 7.0 \\
C_3 &= 6.0 \\
C_4 &= 8.0 \\
C_5 &= 5.5 \\
C_6 &= 9.0 \\
C_7 &= 6.5 \\
C_8 &= 7.5 \\
C_9 &= 8.2 \\
C_{10} &= 5.7 \\
\end{align*}
\]

2. **Minimum Allocation of First Product (Smartphone):**
   \[
   \sum_{i=1}^{10} x_{i1} \geq 5
   \]

3. **Nonnegativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
   \]

---

### Complete Model (Explicitly)

\[
\begin{align*}
\max\quad & \sum_{i=1}^{10} \Big(200\, x_{i1} + 1500\, x_{i2} + 100\, x_{i3} + 800\, x_{i4} + 250\, x_{i5} + 600\, x_{i6} + 150\, x_{i7} + 80\, x_{i8} + 50\, x_{i9} + 300\, x_{i10} \\
&\qquad + 400\, x_{i11} + 120\, x_{i12} + 60\, x_{i13} + 40\, x_{i14} + 30\, x_{i15} + 25\, x_{i16} + 100\, x_{i17} + 500\, x_{i18} + 90\, x_{i19} + 180\, x_{i20} \Big) \\
\text{s.t.}\quad & \sum_{j=1}^{20} w_j x_{1j} \leq 5.0 \\
& \sum_{j=1}^{20} w_j x_{2j} \leq 7.0 \\
& \sum_{j=1}^{20} w_j x_{3j} \leq 6.0 \\
& \sum_{j=1}^{20} w_j x_{4j} \leq 8.0 \\
& \sum_{j=1}^{20} w_j x_{5j} \leq 5.5 \\
& \sum_{j=1}^{20} w_j x_{6j} \leq 9.0 \\
& \sum_{j=1}^{20} w_j x_{7j} \leq 6.5 \\
& \sum_{j=1}^{20} w_j x_{8j} \leq 7.5 \\
& \sum_{j=1}^{20} w_j x_{9j} \leq 8.2 \\
& \sum_{j=1}^{20} w_j x_{10j} \leq 5.7 \\
& \sum_{i=1}^{10} x_{i1} \geq 5 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,10;\ j=1,\ldots,20 \\
\end{align*}
\]

Where \( w_j \) is the weight of product \( j \) as listed above.

---

**All coefficients, identifiers, and constraints are mapped directly from the provided CSV data and user requirements.**