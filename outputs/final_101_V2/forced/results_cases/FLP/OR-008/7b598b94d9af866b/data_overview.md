Here is the complete data retrieved from the three files, preserving all identifiers, values, and source-row positions:

---

**customer_demand.csv**

| Row | Customers  | Demand |
|-----|------------|--------|
| 1   | Customer1  | 70     |
| 2   | Customer2  | 80     |
| 3   | Customer3  | 60     |
| 4   | Customer4  | 90     |
| 5   | Customer5  | 85     |
| 6   | Customer6  | 95     |

---

**supply_capacity.csv**

| Row | Suppliers  | Supply_Capacity |
|-----|------------|-----------------|
| 1   | Supplier1  | 200             |
| 2   | Supplier2  | 250             |
| 3   | Supplier3  | 230             |
| 4   | Supplier4  | 220             |
| 5   | Supplier5  | 210             |

---

**transportation_costs.csv**

| Row | Supplier   | Customer1 | Customer2 | Customer3 | Customer4 | Customer5 | Customer6 |
|-----|------------|-----------|-----------|-----------|-----------|-----------|-----------|
| 1   | Supplier1  | 2         | 3         | 1         | 2         | 3         | 2         |
| 2   | Supplier2  | 1         | 2         | 3         | 2         | 3         | 2         |
| 3   | Supplier3  | 3         | 1         | 2         | 3         | 2         | 3         |
| 4   | Supplier4  | 2         | 3         | 2         | 1         | 3         | 4         |
| 5   | Supplier5  | 3         | 2         | 3         | 3         | 2         | 3         |

---

**Preserved Axis and Source Orientation:**

- **Facilities (Warehouses):** Supplier1, Supplier2, Supplier3, Supplier4, Supplier5 (rows in supply_capacity.csv and transportation_costs.csv)
- **Customers (Stores):** Customer1, Customer2, Customer3, Customer4, Customer5, Customer6 (columns in customer_demand.csv and transportation_costs.csv)
- **FixedCost:** Not provided in the data.
- **Capacity:** As per supply_capacity.csv, matched by Supplier ID.
- **Demand:** As per customer_demand.csv, matched by Customer ID.
- **Cost Matrix:** transportation_costs.csv, with rows as Suppliers and columns as Customers, shape (5,6).

No data has been transposed, truncated, padded, or inferred beyond the original files. All axes and identifiers are preserved as in the source.