from sinan import people

print "-----"

days_of_week = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
for day in days_of_week:
	print "the day is " + day

for person in people:
	print type(person)
	print str(person['age']) + " and lives in " + person['location']