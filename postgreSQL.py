import psycopg2
import main



con = psycopg2.connect(
    database="TestingDataBase",
    user="postgres",
    password="axdzm3zp9",
    host="localhost",
    port=5432
)

cursor_obj = con.cursor()
#cursor_obj.execute("CREATE TABLE person (id BIGINT NOT NULL PRIMARY KEY, first_name VARCHAR(50))")
#cursor_obj.execute("INSERT INTO person (id, first_name) VALUES ('1', 'Sally')")
#cursor_obj.execute("SELECT * FROM person")

#result = cursor_obj.fetchall()

#cursor_obj.execute("INSERT INTO PERSON(id, first_name) VALUES (2, 'Bob')")

#insert_script = "INSERT INTO person (id, first_name) VALUES (3, 'Hi')"
#cursor_obj.execute(insert_script)

#create_table_script = "CREATE TABLE car (id BIGSERIAL NOT NULL PRIMARY KEY, model VARCHAR(30))"
#cursor_obj.execute(create_table_script)


insert_element = "INSERT INTO car (id, model) VALUES (87, 'Toyota')"
cursor_obj.execute(insert_element)



con.commit()

cursor_obj.close()
con.close()

#print(result)