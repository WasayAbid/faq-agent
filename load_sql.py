# create_database.py

import sqlite3

# --- 1. ESTABLISH CONNECTION ---
# This will create the 'dubai_faq.db' file in your project folder
try:
    conn = sqlite3.connect('dubai_faq.db')
    cursor = conn.cursor()
    print("Database connection established successfully.")

    # --- 2. CREATE TABLE ---
    # We use "IF NOT EXISTS" so the script can be run multiple times without error
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS faqs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        category TEXT
    )
    """)
    print("Table 'faqs' created or already exists.")

    # --- 3. INSERT DATA ---
    # This is sample data. These are clear, factual questions
    # that are perfect for an SQL database.
    faqs_to_insert = [
        (
            "How tall is the Burj Khalifa?",
            "The Burj Khalifa stands at a height of 828 meters (2,717 feet).",
            "Landmarks"
        ),
        (
            "What is the currency of Dubai?",
            "The currency used in Dubai is the UAE Dirham, commonly abbreviated as AED.",
            "General Info"
        ),
        (
            "How long is a standard tourist visa for Dubai?",
            "A standard tourist visa for Dubai is typically valid for 30 or 60 days, though options can vary based on nationality.",
            "Visas"
        ),
        (
            "What year did the Dubai Metro open?",
            "The Dubai Metro officially opened to the public on September 9, 2009.",
            "Transportation"
        )
    ]

    # Insert data only if the table is empty to avoid duplicates
    cursor.execute("SELECT COUNT(*) FROM faqs")
    count = cursor.fetchone()[0]
    if count == 0:
        cursor.executemany(
            "INSERT INTO faqs (question, answer, category) VALUES (?, ?, ?)",
            faqs_to_insert
        )
        print(f"{len(faqs_to_insert)} new records inserted into 'faqs' table.")
    else:
        print("Data already exists in 'faqs' table, skipping insertion.")


    # --- 4. COMMIT AND CLOSE ---
    conn.commit()
    print("Changes committed to the database.")

except sqlite3.Error as e:
    print(f"Database error: {e}")

finally:
    if conn:
        conn.close()
        print("Database connection closed.")