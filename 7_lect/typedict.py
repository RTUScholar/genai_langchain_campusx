from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    email:str

new_person : Person = {'name':'priyanshu', 'age':24, 'email':'priyanshu@example.com'}    
print(new_person)