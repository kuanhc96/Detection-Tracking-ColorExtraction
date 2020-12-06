import sqlite3
from ColorLabeler import ColorLabeler as CL

conn = sqlite3.connect('bounding_boxes.db')
conn2 = sqlite3.connect('bbox_rgb.db')

c = conn.cursor()
c2 = conn.cursor()


try:
    c2.execute("""CREATE TABLE bbox_colors (
                id INTEGER,
                R REAL,
                G REAL,
                B REAL,
                color TEXT
                )""")
except:
    print("table created")

c.execute("SELECT id, AVG(red) as R, AVG(green) as G, AVG(blue) as B "
              "FROM bounding_boxes "
              "GROUP BY id ")

rgb_rows = c.fetchall()

cl = CL()

for row in rgb_rows:
    if (row[1] != None and row[2] != None and row[3] != None):
        rgb = row[1:]
        color = cl.match_color(rgb)
        c2.execute("INSERT INTO bbox_colors VALUES (?, ?, ?, ?, ?)", (row[0], row[1], row[2], row[3], color))

c2.execute("SELECT * FROM bbox_colors")
for row in c2.fetchall():
    print(row)

