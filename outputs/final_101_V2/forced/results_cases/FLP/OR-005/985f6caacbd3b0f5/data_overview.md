Here is the complete retrieval of all data from the three files, preserving all identifiers, values, and matrix axes as requested:

---

### customer_demand.csv

| Customers | demand |
|-----------|--------|
| demand1   | 9      |
| demand2   | 66     |
| demand3   | 56     |
| demand4   | 17     |
| demand5   | 43     |
| demand6   | 62     |
| demand7   | 10     |
| demand8   | 37     |

---

### supply_capacity.csv

| Supplier   | supply_capacity |
|------------|----------------|
| supplier1  | 60             |
| supplier2  | 22             |
| supplier3  | 16             |
| supplier4  | 14             |
| supplier5  | 19             |
| supplier6  | 70             |
| supplier7  | 60             |
| supplier8  | 39             |

---

### transportation_costs.csv

Each row corresponds to a supply facility (distribution center), and each column to a customer group. The values are the transportation cost per unit from the given supplier to the given customer.

|            | demand1 | demand2 | demand3 | demand4 | demand5 | demand6 | demand7 | demand8 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|
| supply1    | 0.0302  | 229.5072| 198.6236| 12.9951 | 211.2073| 134.9443| 9.8222  | 11.3941 |
| supply2    | 232.3469| 3.6259  | 0.2861  | 45.7313 | 2.8305  | 107.0589| 299.9632| 23.7994 |
| supply3    | 11.0619 | 0.2042  | 0.2789  | 45.7219 | 59.5490 | 5.0975  | 300.0012| 23.7113 |
| supply4    | 235.1795| 43.7947 | 40.7098 | 0.0777  | 4.2377  | 131.7092| 296.5559| 29.8109 |
| supply5    | 211.8581| 47.6018 | 50.0401 | 86.1455 | 0.0620  | 5.3346  | 270.0629| 3.8539  |
| supply6    | 6.4551  | 88.1632 | 5.0471  | 151.4612| 5.2908  | 0.0460  | 9.9367  | 103.7546|
| supply7    | 174.2723| 250.5822| 253.9041| 16.2355 | 12.6431 | 175.0673| 2.9838  | 317.0655|
| supply8    | 207.8701| 1.5172  | 24.0272 | 27.1340 | 73.2067 | 125.7291| 15.4631 | 0.2016  |

- **Row axis:** supply1, supply2, supply3, supply4, supply5, supply6, supply7, supply8 (distribution centers)
- **Column axis:** demand1, demand2, demand3, demand4, demand5, demand6, demand7, demand8 (customer groups)
- **Values:** Transportation cost per unit

---

**All identifiers, values, and matrix axes are preserved as in the original data.**