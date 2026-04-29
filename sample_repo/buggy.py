def divide(a, b):
    # bug: no check for division by zero
    return a / b

def get_first_item(lst):
    # bug: no check if list is empty
    return lst[0]

def read_file(path):
    # bug: file never closed
    f = open(path, 'r')
    return f.read()

def calculate_average(numbers):
    # bug: no check for empty list
    total = sum(numbers)
    return total / len(numbers)