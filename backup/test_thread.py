import queue
import threading
import multiprocessing as mp


def foo(s):
    dicionario = []
    dicionario.append('a')
    return dicionario

que = queue.Queue()
threads_list = list()

#print(lambda a: a * 2)

print('Cores: ', mp.cpu_count())


for i in range(mp.cpu_count()):
    t = threading.Thread(target=lambda q, arg1: q.put(foo(arg1)), args=(que, 'world!'))
    t.start()
    threads_list.append(t)

# Join all the threads
for t in threads_list:
    t.join()

list_geral = []
# Check thread's return value
while not que.empty():
    result = que.get()
    list_geral.append(result)

print(list_geral)
#t2 = Thread(target=lambda q, arg1: q.put(foo(arg1)), args=(que, 'world!'))
#t2.start()
#threads_list.append(t2)



