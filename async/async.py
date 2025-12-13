import os, sys, json, asyncio, time, threading

#print("START")
print("imported async utils")
print("Available CPUs:", os.cpu_count())
async def wait(seconds):
    await asyncio.sleep(seconds)


async def simple(text, sleep=False):
    print(text)
    if sleep:
        await wait(5)
    print(text)




async def test():
    print("Async function")
    
    t1 = simple("fun 1", True)
    t2 = simple("fun 2")
    await asyncio.gather(t1, t2)
    print("Async done")


#asyncio.run(test())

#print("DONE")





class AsyncPool(object):
    def __init__(self, n_workers="auto"):
        self.pending = {}
        self.done = {}
        self.started = False
        self.start_time = None
        self.end_time = None

    def __len__(self):
        return len(self.pending) + len(self.done)

    def __repr__(self):
        return f"<AsyncPool: pending:{len(self.pending)} done:{self.done}>"

    def __add__(self, awaitable):
        self.pending[len(self)] = asyncio.createTask(awaitable)

    async def start():
        asyncio.gather()



pool = AsyncPool()
print(len(pool))
print(pool)


