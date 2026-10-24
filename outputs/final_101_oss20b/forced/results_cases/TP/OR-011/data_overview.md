{
  "query": "The supermarket offers a variety of top-selling products, with associated revenue data provided in the ‘Revenue’ column. Each product has its own demand level during the sales horizon. The company’s objective is to maximize total revenue by allocating the available inventory of products classified under ‘id999’. The initial inventory levels for the ‘id999’ products are detailed in the ‘Initial Inventory’ column. During the sales horizon, no restocking is allowed, and there are no in-transit inventories.\n\nDemand for each product during the sales period is assumed to be deterministic and known in advance, with demand quantities specified in the ‘Demand’ column. The decision variables x_i represent the number of units of each ‘id999’ product i that the company plans to fulfill, where each x_i is a non-negative integer. Because fulfilled orders cannot exceed either the available inventory or the realized demand, the fulfillment quantities must satisfy both inventory and demand constraints.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "id_number",
        "Revenue",
        "Demand",
        "Initial Inventory"
      ],
      "file_name": "OnlineRetailSalesDataset.csv",
      "records": [
        {
          "data_row_number": 899,
          "values": {
            "Demand": "8171",
            "Initial Inventory": "56450",
            "Revenue": "434.74",
            "id_number": "id999"
          }
        }
      ],
      "table_id": "file_0_view_0"
    }
  ]
}
