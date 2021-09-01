import json
person = '{"name": "Bob", "languages": ["English", "Fench"]}'

#it should be less than 1 sec, much faster than c++ boost..
for i in range(10000):
    person_dict = json.loads(person)
    if i%1000==0:
        print(person_dict["name"])