Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$.

Let $i$ index shelves (displays) with ShelfID from the retrieved data, and $j$ index products with ProductName from the retrieved data.

Let $c_i$ be the Capacity of shelf $i$.

Let $v_j$ be the Value of product $j$.

Let $w_j$ be the Weight of product $j$.

Let the first product in the source order be "Smartphone".

The model is:

Objective:
\[
\max \sum_{i \in \{1,2,3,4,5,6,7,8,9,10\}} \sum_{j \in \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Monitor}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}} v_j x_{ij}
\]

Subject to:

For each shelf $i$ (with ShelfID as below):

\[
\sum_{j} w_j x_{ij} \leq c_i \qquad \forall i
\]

That is, for each shelf:

- ShelfID 1: $\sum_{j} w_j x_{1j} \leq 5.0$
- ShelfID 2: $\sum_{j} w_j x_{2j} \leq 7.0$
- ShelfID 3: $\sum_{j} w_j x_{3j} \leq 6.0$
- ShelfID 4: $\sum_{j} w_j x_{4j} \leq 8.0$
- ShelfID 5: $\sum_{j} w_j x_{5j} \leq 5.5$
- ShelfID 6: $\sum_{j} w_j x_{6j} \leq 9.0$
- ShelfID 7: $\sum_{j} w_j x_{7j} \leq 6.5$
- ShelfID 8: $\sum_{j} w_j x_{8j} \leq 7.5$
- ShelfID 9: $\sum_{j} w_j x_{9j} \leq 8.2$
- ShelfID 10: $\sum_{j} w_j x_{10j} \leq 5.7$

Minimum total quantity of the first product ("Smartphone") across all shelves:

\[
\sum_{i=1}^{10} x_{i,\text{Smartphone}} \geq 5
\]

Nonnegativity and integrality:

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

Where:

- ShelfIDs: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- Products (in source order):

    1. Smartphone (Value: 200, Weight: 1.0)
    2. Laptop (1500, 5.0)
    3. Headphones (100, 0.5)
    4. Camera (800, 2.0)
    5. Smartwatch (250, 0.3)
    6. Tablet (600, 1.5)
    7. Bluetooth Speaker (150, 1.0)
    8. Keyboard (80, 0.8)
    9. Mouse (50, 0.2)
    10. Monitor (300, 3.0)
    11. Printer (400, 4.0)
    12. External Hard Drive (120, 0.5)
    13. Router (60, 0.3)
    14. Power Bank (40, 0.4)
    15. Memory Card (30, 0.05)
    16. USB Flash Drive (25, 0.02)
    17. Smart Home Hub (100, 0.6)
    18. Gaming Console (500, 4.0)
    19. Fitness Tracker (90, 0.2)
    20. E-Reader (180, 0.5)

All coefficients and identifiers are as retrieved.