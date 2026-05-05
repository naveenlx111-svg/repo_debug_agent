def divide(a, b):
    # check for division by zero
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b

def get_first_item(lst):
    # bug: no check if list is empty
    return lst[0]

def read_file(path):
    # bug: file never closed
    f = open(path, 'r')
    return f.read()

def calculate_average(numbers):
    if not numbers:
        raise ValueError("Cannot calculate average of an empty list")
    if not all(isinstance(x, (int, float)) for x in numbers):
        raise TypeError("All numbers must be integers or floats")
    total = sum(numbers)
    return total / len(numbers)