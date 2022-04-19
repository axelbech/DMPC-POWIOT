from pickle import loads, dumps
from test2 import myClass

if __name__ == '__main__':
    c = myClass()
    e = myClass()
    d = dumps(c)
    d = dumps(c)
    l = loads(d)
    # c.execute_actions()
    # print(loads(dumps(s0)))
    # with ProcessPoolExecutor() as executor:
    #     future = executor.submit(c.get_action,s0.master,lbw,ubw)
    #     print(future.result())

# print(c.get_action(s0, lbw, ubw))

#does work
# print(loads(dumps(c.s_num)))