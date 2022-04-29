import multiprocessing
from multiprocessing import Pipe, Queue, Process
import os
from os import dup
import random
from re import sub
import time


class Coordinator():
    def __init__(self, connections: dict, receive_queue) -> None:
        self.number = 0
        self.connections = connections
        self.receive_queue = receive_queue
        self.gifts = {name: 0 for name in self.connections}
    
    def important_calculation(self):
        self.number += random.randint(-1, 1)
        time.sleep(1)
        print(f'number now equals {self.number}, pid = {os.getpid()}')
            
    def run_loop(self):
        while True:
            self.important_calculation()
            for controller_conn in self.connections.values():
                controller_conn.send(self.number)
            while not self.receive_queue.empty():
                val = self.receive_queue.get()
                print(f'\nReceived value = {val}')
                self.gifts[val[0]] = val[1]
                
class Controller():
    def __init__(self, name, input_connection, queue) -> None:
        self.name = name
        self.connection = input_connection
        self.queue = queue
    
    def run_loop(self):
        while True:
            if self.connection.poll():
                value = self.connection.recv()
                value_aug = self.do_something(value)
                self.queue.put((self.name, value_aug))
            time.sleep(1)
            
    def do_something(self, value):
        randint = random.randint(-10, 10)
        print(f'\nvalue={value}, adding {randint}, pid = {os.getpid()}')
        return value + randint
    
    
if __name__ == '__main__':
    submit_queue = Queue()
    conn_recv, conn_send = Pipe(duplex=False)
    controller = Controller(name='axel', input_connection=conn_recv, queue=submit_queue)
    conn_dict = dict(axel = conn_send)
    coordinator = Coordinator(connections=conn_dict, receive_queue=submit_queue)
    
    p_coordinator = Process(
        target=Coordinator.run_loop, 
        args=(coordinator,)
        )
    p_controller = Process(
        target=Controller.run_loop, 
        args=(controller,)
        )
    p_coordinator.start()
    p_controller.start()
    # print(q.get())    # prints "[42, None, 'hello']"
    # p.join()
    # time.sleep(10)
    # p_controller.terminate()
    # p_coordinator.terminate()
    # time.sleep(1)
    # p_controller.close()
    # p_coordinator.close()