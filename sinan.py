sinan = { "age": 23,
"location": "baltimore",
"gender": "male",
"occupation":"Professor"}


denuz = {"age": 29,
"location": "washington",
"gender":"female",
'occupation':'security'}

people = [sinan,denuz]
if __name__ == "__main__":
	print sinan['age']
	print sinan['location']
	print people[0]
	print people[0]['age']