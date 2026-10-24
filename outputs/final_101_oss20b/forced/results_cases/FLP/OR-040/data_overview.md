{
  "query": "A real-estate developer is planning property development in New York City. For each area (e.g., Queens, Brooklyn, etc.), the developer has a “products.csv” file that records the benefit coefficient for that area.\n    The developer also faces a single overall development-capacity constraint, provided in “capacity.csv.” file. The objective is to decide the daily scale of development in each area so as to maximize total benefit while ensuring that the sum of all development units does not exceed the overall capacity. The decision variables  x_i  represent the scale of development in area  i  each day.The decision variables must be integers.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "Capacity"
      ],
      "file_name": "capacity.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Capacity": "4466"
          }
        }
      ],
      "table_id": "file_0_view_0"
    },
    {
      "columns": [
        "ProductName",
        "Value"
      ],
      "file_name": "products.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "ProductName": "Queens",
            "Value": "443"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "ProductName": "Brooklyn",
            "Value": "522"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "ProductName": "Manhattan",
            "Value": "300"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "ProductName": "Bronx",
            "Value": "767"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "ProductName": "Staten Island",
            "Value": "300"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "ProductName": "Harlem",
            "Value": "309"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "ProductName": "Upper East Side",
            "Value": "598"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "ProductName": "Lower Manhattan",
            "Value": "460"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "ProductName": "Midtown",
            "Value": "318"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "ProductName": "Long Island City",
            "Value": "126"
          }
        },
        {
          "data_row_number": 10,
          "values": {
            "ProductName": "Williamsburg",
            "Value": "593"
          }
        },
        {
          "data_row_number": 11,
          "values": {
            "ProductName": "Bushwick",
            "Value": "871"
          }
        },
        {
          "data_row_number": 12,
          "values": {
            "ProductName": "Flatbush",
            "Value": "858"
          }
        },
        {
          "data_row_number": 13,
          "values": {
            "ProductName": "Greenpoint",
            "Value": "321"
          }
        },
        {
          "data_row_number": 14,
          "values": {
            "ProductName": "Park Slope",
            "Value": "275"
          }
        },
        {
          "data_row_number": 15,
          "values": {
            "ProductName": "Astoria",
            "Value": "700"
          }
        },
        {
          "data_row_number": 16,
          "values": {
            "ProductName": "Jackson Heights",
            "Value": "685"
          }
        },
        {
          "data_row_number": 17,
          "values": {
            "ProductName": "Flushing",
            "Value": "940"
          }
        },
        {
          "data_row_number": 18,
          "values": {
            "ProductName": "Sunnyside",
            "Value": "522"
          }
        },
        {
          "data_row_number": 19,
          "values": {
            "ProductName": "Ditmars",
            "Value": "763"
          }
        }
      ],
      "table_id": "file_1_view_0"
    }
  ]
}
