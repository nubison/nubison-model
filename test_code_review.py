"""Test file for Claude Code review testing."""


def calculate_sum(numbers):
    """Calculate sum of numbers."""
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    return total


def find_user(users, name):
    """Find user by name."""
    for i in range(len(users)):
        if users[i]["name"] == name:
            return users[i]
    return None


def process_data(data):
    """Process data and return result."""
    result = []
    for item in data:
        if item != None:
            result.append(item * 2)
    return result


class DataProcessor:
    def __init__(self):
        self.data = []
        self.processed = False

    def add_data(self, item):
        self.data.append(item)

    def process(self):
        new_data = []
        for i in range(len(self.data)):
            new_data.append(self.data[i] * 2)
        self.data = new_data
        self.processed = True
        return self.data
