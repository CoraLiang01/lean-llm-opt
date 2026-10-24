{
  "query": "A factory produces three products, I, II, and III. Each product goes through two processing procedures, A and B. The factory has two types of equipment, A1 and A2, to complete procedure A, and three types of equipment, B1, B2, and B3, to complete procedure B. Product I can be processed on either type of A equipment or any type of B equipment. Product II can be processed on any type of A equipment, but when completing procedure B, it can only be processed on B1 equipment. Product III can only be processed on A2 and B2 equipment. Given the processing time, raw material cost, product selling price, available equipment operating time, and equipment cost at full load for each type of equipment, as shown in 43.csv, determine the optimal production plan to maximize profit. All production quantities should be continuous.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "Equipment / Cost",
        "Product I",
        "Product II",
        "Product III",
        "Available Equipment Operating Time",
        "Equipment Cost at Full Load (yuan)"
      ],
      "file_name": "43.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Available Equipment Operating Time": "6000",
            "Equipment / Cost": "A1",
            "Equipment Cost at Full Load (yuan)": "300",
            "Product I": "5",
            "Product II": "10",
            "Product III": ""
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Available Equipment Operating Time": "10000",
            "Equipment / Cost": "A2",
            "Equipment Cost at Full Load (yuan)": "321",
            "Product I": "7",
            "Product II": "9",
            "Product III": "12"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Available Equipment Operating Time": "4000",
            "Equipment / Cost": "B1",
            "Equipment Cost at Full Load (yuan)": "250",
            "Product I": "6",
            "Product II": "8",
            "Product III": ""
          }
        }
      ],
      "table_id": "file_0_view_0"
    }
  ]
}
