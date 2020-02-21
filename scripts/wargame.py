import time
import timeit
from collections import deque
from random import shuffle
from joblib import Parallel, delayed
import multiprocessing

def deck(n):
    for i in range(4*n):
        yield (i // 4) + 1 

def shuffled_deck(n):
    d = list(deck(n))
    shuffle(d)
    return d

def split_deck(deck):
    index = int(len(deck)/2)
    # cut the deck in 2 equal parts
    g0 = tuple(deck[:index])
    g1 = tuple(deck[index:])

    # we don't take into account which player wins or looses
    return (g0,g1) if (g0<g1) else (g1,g0)

def enumerate_games(n):
    counts = [4] * n
    game = [None]*(4*n)
    
    def recurse(index, hashes):
         for value in range(n): # each card can have a value up to n
            
            if counts[value] == 0: # no more card with this value
                continue
            
            counts[value] -= 1
            
            game[index] = value+1
            
            if index == 4*n - 1:
                g = split_deck(game)
                
                # check that this distribution was not already returned
                h = hash(g)
                if not h in hashes:
                    hashes.add(h)
                    yield g
                
            else:
                yield from recurse(index+1, hashes)
            
            counts[value] += 1
       
    yield from recurse(0, set())

def play_game(packs):
    playerA = deque(packs[0])
    playerB = deque(packs[1])
    stack = deque()
    tricks = 0
    cards = 0
    hashes = []
    
    
    
    #print(packs)
    
    while (len(playerA) and len(playerB)):
        
        if not stack:
            h = hash(tuple(list(playerA) + [0] + list(playerB)))
            if h in hashes:
                #print(hashes, h)
                offset = hashes.index(h)
                period = len(hashes) - offset
                return packs, 3, offset, period

            hashes += [h]
        
        a = playerA.popleft()
        b = playerB.popleft()
        #print(f'  a:{a}, b:{b}')

        cards += 1
        if not stack: tricks += 1    

        if (a > b):
            playerA.extend([a,b])
            playerA.extend(stack)
            stack.clear()
            #print(f'A:{playerA}, B:{playerB}')
        elif (a < b):
            playerB.extend([b,a])
            playerB.extend(stack)
            stack.clear()
            #print(f'A:{playerA}, B:{playerB}')
        else:
            stack.extendleft([a,b])
            #print(f'    stack:{stack}')
    
    winner = 1 if len(playerA) \
        else 2 if len(playerB) \
        else 0
    
    return packs, winner, tricks, cards
    
# if __name__ == '__main__':
#     # winner_name = ['None','A','B','Infinite']
#     # game, winner, tricks, cards = play_game( split_deck(shuffled_deck(3)) )
#     # print(game, winner_name[winner], tricks, cards)

#     num_cores = multiprocessing.cpu_count()
#     print (num_cores)

#     print("sequential")
#     print(timeit.timeit('[play_game(game) for game in enumerate_games(3)]', globals=globals(), number=10))

#     time.sleep(2)
    
#     # print("loky")
#     # print(timeit.timeit('Parallel(n_jobs=16, backend = "loky", batch_size=1000)(delayed(play_game)(game) for game in enumerate_games(3))', globals=globals(), number=10))
    
#     time.sleep(2)
    
#     for batch_size in range(100, 2000, 100):
#         print(f'threading batch {batch_size}')
#         print(timeit.timeit(f'Parallel(n_jobs=-1, backend = "threading", batch_size={batch_size})(delayed(play_game)(game) for game in enumerate_games(3))', globals=globals(), number=5))

from threading import Thread
from queue import Queue
from itertools import islice, chain

MAX_GAMES = 100000

def batch(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip(*args)

def run_in_sequential(n):
    results = []
    for item in islice(enumerate_games(n),MAX_GAMES):
        game, winner, tricks, cards = play_game(item)
        results += [{'game':game, 'winner':winner, 'tricks':tricks, 'cards':cards}]
    return results
   
def run_in_parallel(num_worker_threads,n,batch_size):

    q = Queue(maxsize = 10000)
    threads = []

    def worker(results):
        while True:
            batch = q.get()
            if batch is None:
                break

            for item in batch:
                game, winner, tricks, cards = play_game(item)
                results += [{'game':game, 'winner':winner, 'tricks':tricks, 'cards':cards}]

            q.task_done()

    all_results = [[]] * num_worker_threads
    for i in range(num_worker_threads):
        t = Thread(target=worker, args=(all_results[i],))
        t.start()
        threads.append(t)

    for games in batch(islice(enumerate_games(n),MAX_GAMES),batch_size):
        q.put(games)

    # block until all tasks are done
    q.join()

    # stop workers
    for i in range(num_worker_threads):
        q.put(None)
    for t in threads:
        t.join()

    #return [item for sublist in all_results for item in sublist]
    return list(chain(*all_results))


#------------------------------------------


# N = 4
# niter = 10
# print("sequential")
# print(timeit.timeit(f'run_in_sequential({N})', globals=globals(), number=niter) / niter)
# for num_worker_threads in [2,4,8,16,32]:
#     print(f'{num_worker_threads} threads')
#     print(timeit.timeit(f'run_in_parallel({num_worker_threads},{N},1000)', globals=globals(), number=niter) / niter)


#------------------------------------------

import concurrent.futures

if __name__ == '__main__':
    N = 4
    niter = 10
    MAX_GAMES = 100000

    # print("sequential")
    # print(timeit.timeit(f'run_in_sequential({N})', globals=globals(), number=niter) / niter)

    for num_worker_threads in [8]: #[2,4,8,16]:
        begin = time.perf_counter()
        print(f'{num_worker_threads} processes')
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker_threads) as executor:
            for _ in range(niter):
                executor.map(play_game, enumerate_games(N), chunksize=1000)

        runtime = time.perf_counter()-begin
        print(f'runtime:{runtime} for {niter} iterations')