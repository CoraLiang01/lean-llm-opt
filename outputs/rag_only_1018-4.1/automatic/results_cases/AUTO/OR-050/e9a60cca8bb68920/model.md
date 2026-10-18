Let:
- \( I = \{1,2,3,4,5,6,7,8,9,10\} \) be the set of ShelfIDs (displays), as given in capacity.csv.
- \( J = \{1,2,\ldots,20\} \) be the set of products, in the order listed in products.csv:
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

Parameters:
- For each shelf \( i \in I \), let \( C_i \) be its capacity:
    - \( C_1 = 5.0 \)
    - \( C_2 = 7.0 \)
    - \( C_3 = 6.0 \)
    - \( C_4 = 8.0 \)
    - \( C_5 = 5.5 \)
    - \( C_6 = 9.0 \)
    - \( C_7 = 6.5 \)
    - \( C_8 = 7.5 \)
    - \( C_9 = 8.2 \)
    - \( C_{10} = 5.7 \)
- For each product \( j \in J \), let \( v_j \) be its value and \( w_j \) its weight:
    1. Smartphone: \( v_1 = 200 \), \( w_1 = 1.0 \)
    2. Laptop: \( v_2 = 1500 \), \( w_2 = 5.0 \)
    3. Headphones: \( v_3 = 100 \), \( w_3 = 0.5 \)
    4. Camera: \( v_4 = 800 \), \( w_4 = 2.0 \)
    5. Smartwatch: \( v_5 = 250 \), \( w_5 = 0.3 \)
    6. Tablet: \( v_6 = 600 \), \( w_6 = 1.5 \)
    7. Bluetooth Speaker: \( v_7 = 150 \), \( w_7 = 1.0 \)
    8. Keyboard: \( v_8 = 80 \), \( w_8 = 0.8 \)
    9. Mouse: \( v_9 = 50 \), \( w_9 = 0.2 \)
    10. Monitor: \( v_{10} = 300 \), \( w_{10} = 3.0 \)
    11. Printer: \( v_{11} = 400 \), \( w_{11} = 4.0 \)
    12. External Hard Drive: \( v_{12} = 120 \), \( w_{12} = 0.5 \)
    13. Router: \( v_{13} = 60 \), \( w_{13} = 0.3 \)
    14. Power Bank: \( v_{14} = 40 \), \( w_{14} = 0.4 \)
    15. Memory Card: \( v_{15} = 30 \), \( w_{15} = 0.05 \)
    16. USB Flash Drive: \( v_{16} = 25 \), \( w_{16} = 0.02 \)
    17. Smart Home Hub: \( v_{17} = 100 \), \( w_{17} = 0.6 \)
    18. Gaming Console: \( v_{18} = 500 \), \( w_{18} = 4.0 \)
    19. Fitness Tracker: \( v_{19} = 90 \), \( w_{19} = 0.2 \)
    20. E-Reader: \( v_{20} = 180 \), \( w_{20} = 0.5 \)

Decision variables:
- For each shelf \( i \in I \) and product \( j \in J \), let \( x_{ij} \) be the number of units of product \( j \) placed on shelf \( i \).
- \( x_{ij} \) are nonnegative integers: \( x_{ij} \in \mathbb{Z}_+, \forall i \in I, j \in J \).

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
\]
That is, maximize the total value of all products placed across all shelves.

Constraints:
1. Shelf capacity constraints (for each shelf \( i \)):
\[
\sum_{j=1}^{20} w_j x_{ij} \leq C_i, \quad \forall i \in I
\]
Explicitly, for each shelf:
- Shelf 1: \( 1.0x_{1,1} + 5.0x_{1,2} + 0.5x_{1,3} + 2.0x_{1,4} + 0.3x_{1,5} + 1.5x_{1,6} + 1.0x_{1,7} + 0.8x_{1,8} + 0.2x_{1,9} + 3.0x_{1,10} + 4.0x_{1,11} + 0.5x_{1,12} + 0.3x_{1,13} + 0.4x_{1,14} + 0.05x_{1,15} + 0.02x_{1,16} + 0.6x_{1,17} + 4.0x_{1,18} + 0.2x_{1,19} + 0.5x_{1,20} \leq 5.0 \)
- Shelf 2: (same as above, with \( x_{2,j} \), \( \leq 7.0 \))
- Shelf 3: (\( x_{3,j} \), \( \leq 6.0 \))
- Shelf 4: (\( x_{4,j} \), \( \leq 8.0 \))
- Shelf 5: (\( x_{5,j} \), \( \leq 5.5 \))
- Shelf 6: (\( x_{6,j} \), \( \leq 9.0 \))
- Shelf 7: (\( x_{7,j} \), \( \leq 6.5 \))
- Shelf 8: (\( x_{8,j} \), \( \leq 7.5 \))
- Shelf 9: (\( x_{9,j} \), \( \leq 8.2 \))
- Shelf 10: (\( x_{10,j} \), \( \leq 5.7 \))

2. Minimum allocation of the first product (Smartphone) across all shelves:
\[
\sum_{i=1}^{10} x_{i,1} \geq 5
\]

3. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_+, \quad \forall i \in I, j \in J
\]

Full Model:
\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t. } & \sum_{j=1}^{20} w_j x_{ij} \leq C_i, \quad \forall i=1,\ldots,10 \\
& \sum_{i=1}^{10} x_{i,1} \geq 5 \\
& x_{ij} \in \mathbb{Z}_+, \quad \forall i=1,\ldots,10,\ j=1,\ldots,20
\end{align*}
\]
where all coefficients and indices are as specified above, using the explicit ShelfID and ProductName order from the supplied data.