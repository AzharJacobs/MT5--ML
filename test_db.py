from data.loader import DatabaseConnection

db = DatabaseConnection()
if not db.connect():
    print("FAIL: could not connect to DB")
    exit(1)

print("Connected OK\n")

tfs = db.get_available_timeframes()
print(f"Timeframes found: {tfs}\n")

for tf in tfs:
    count = db.get_record_count(tf)
    dr = db.get_date_range(tf)
    print(f"  {tf:>6}  rows={count:>8,}  {dr['min_date']} -> {dr['max_date']}")

db.disconnect()
print("\nDone")
