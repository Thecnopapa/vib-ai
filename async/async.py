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


def awaitable(fun, *args, **kwargs):
    afun = async fun
    return afun

def non_awaitable():
    pass

print(non_awaitable, type(non_awaitable))

yes_awaitable = awaitable(non_awaitable)
print(yes_awaitable, type(yes_awaitable))



