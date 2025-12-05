from pydantic import BaseModel, Field

class Student(BaseModel):
    name: str
    age: int = Field(..., gt=0, lt=150)  


new_student = Student(name='priyanshu', age=24)
print(new_student)
print(f"Name: {new_student.name}, Age: {new_student.age}")