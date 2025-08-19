import duckdb

# Path to your DB file
DB_PATH = "ProjectMain/db/data.duckdb"

con = duckdb.connect(DB_PATH)

# List tables
print("Tables:", con.execute("SHOW TABLES").fetchall())

# Look at the first few rows
df = con.execute("SELECT * FROM LoadForecastPivot LIMIT 10").df()
print(df)

# Example: filter by date
df2 = con.execute("""
    SELECT *
    FROM LoadForecastPivot
    WHERE DATE(OperatingDTM) = '2025-08-16'
    ORDER BY Interval
""").df()

print(df2)

con.close()
