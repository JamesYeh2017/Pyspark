from pyhive import hive
cursor = hive.connect('localhost').cursor()
cursor.execute('SELECT * FROM test LIMIT 10')
for c in cursor.fetchall():
    print(c[1])