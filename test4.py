from multiprocessing import Process, Manager
import os

N = 10

def append_to_dict(managed_dict):
    print(f'Changing dictionary, pid = {os.getpid()}')
    managed_dict['hello'] = 'world'
    managed_dict['foo'] = 'bar'
    
def print_func():
    print('hello world')
    
if __name__ == '__main__':
    manager = Manager()
    managed_dict = manager.dict()
    managed_dict['dual_variables'] =  [0] * N
    p = Process(target=append_to_dict, args=(managed_dict,))
    p.start()
    p.join()
    print(f'main pid = {os.getpid()}')
    print(managed_dict)