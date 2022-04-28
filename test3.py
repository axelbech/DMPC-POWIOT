import multiprocessing
from multiprocessing import connection
import random
import time


class Coordinator():
    def __init__(self, connections: dict, receive_queue) -> None:
        self.number = 0
        connections = connections
        self.receive_queue = receive_queue
    
    def important_calculation(self):
        for _ in range(10000):
            self.number += random.randint(-1, -1)
            
    def run_loop(self):
        pass

class Wrapper():
    def __init__(self, connection) -> None:
        self.connection = connection
    
    def run_loop(self):
        while True:
            if self.connection.poll():
                value = self.connection.recv()
                value_aug = self.do_something(value)
                self.connection.send(value_aug)
            
    def do_something(value):
        randint = random.randint(-10, 10)
        print(f'value={value}, adding {randint}')
        return value + randint