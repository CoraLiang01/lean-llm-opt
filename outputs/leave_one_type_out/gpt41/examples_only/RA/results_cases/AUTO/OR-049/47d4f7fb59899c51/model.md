Let:
- S = {1, 2, ..., 10} be the set of shelves, indexed by i (from capacity.csv, ShelfID).
- P = {1, 2, ..., 20} be the set of products, indexed by j (from products.csv, row order).
- Let ProductName_j, Value_j, Weight_j be the name, value, and weight of product j (from products.csv, in order).
- Let Capacity_i be the capacity of shelf i (from capacity.csv, in order).

Decision variables:
x_ij = number of units of product j placed on shelf i, for all i in S, j in P.
x_ij ∈ {0, 1, 2, ...} (nonnegative integers)

Model:

Maximize total value:
maximize
∑_{i∈S} ∑_{j∈P} Value_j * x_ij

where:
- Value_j is as follows (in order of products.csv):
  1. Smartphone: 200
  2. Laptop: 1500
  3. Headphones: 100
  4. Camera: 800
  5. Smartwatch: 250
  6. Tablet: 600
  7. Bluetooth Speaker: 150
  8. Keyboard: 80
  9. Mouse: 50
  10. Monitor: 300
  11. Printer: 400
  12. External Hard Drive: 120
  13. Router: 60
  14. Power Bank: 40
  15. Memory Card: 30
  16. USB Flash Drive: 25
  17. Smart Home Hub: 100
  18. Gaming Console: 500
  19. Fitness Tracker: 90
  20. E-Reader: 180

Subject to shelf capacity constraints:
For each shelf i ∈ S (ShelfID from capacity.csv):

∑_{j∈P} Weight_j * x_ij ≤ Capacity_i

where:
- Weight_j is as follows (in order of products.csv):
  1. Smartphone: 1.0
  2. Laptop: 5.0
  3. Headphones: 0.5
  4. Camera: 2.0
  5. Smartwatch: 0.3
  6. Tablet: 1.5
  7. Bluetooth Speaker: 1.0
  8. Keyboard: 0.8
  9. Mouse: 0.2
  10. Monitor: 3.0
  11. Printer: 4.0
  12. External Hard Drive: 0.5
  13. Router: 0.3
  14. Power Bank: 0.4
  15. Memory Card: 0.05
  16. USB Flash Drive: 0.02
  17. Smart Home Hub: 0.6
  18. Gaming Console: 4.0
  19. Fitness Tracker: 0.2
  20. E-Reader: 0.5

- Capacity_i is as follows (from capacity.csv, in order of ShelfID):
  1: 5.0
  2: 7.0
  3: 6.0
  4: 8.0
  5: 5.5
  6: 9.0
  7: 6.5
  8: 7.5
  9: 8.2
  10: 5.7

Variable domains:
x_ij ∈ {0, 1, 2, ...} for all i ∈ S, j ∈ P

Summary:
maximize ∑_{i=1}^{10} ∑_{j=1}^{20} Value_j * x_ij

subject to for each i = 1,...,10:
  ∑_{j=1}^{20} Weight_j * x_ij ≤ Capacity_i

  x_ij ∈ {0, 1, 2, ...} for all i, j

All coefficients and identifiers are as provided in the CSVs.