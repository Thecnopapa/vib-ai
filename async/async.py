import os, sys, json, asyncio, time, threading, bioiain

#print("START")
print("imported async utils")
print("Available CPUs:", os.cpu_count())
async def wait(seconds):
    print("waiting:", seconds)
    await asyncio.sleep(seconds)


async def simple(text, sleep=False):
    print("start:", text)
    if sleep:
        await wait(1)
        #raise Exception("misc error")
    print("end:", text)




async def test():

    t1 = simple("fun 1", True)
    t2 = simple("fun 2")
    return await asyncio.gather(t1, t2)


#asyncio.run(test())

#print("DONE")





class AsyncPool(object):
    def __init__(self, n_workers="auto"):
        self.tasks = {}
        self.started = False
        self.start_time = None
        self.end_time = None

    def __len__(self):
        return len(self.tasks)

    def __repr__(self):
        pending = 0
        running = 0
        done = 0
        errors = 0
        for task in self.tasks.values():
            if task["status"] == "running":
                running += 1
            elif task["status"] == "done":
                done += 1
            elif task["status"] == "error":
                errors += 1
            elif task["status"] == "pending":
                pending += 1
        return f"<AsyncPool: pending:{pending} running: {running} done:{done} errors:{errors}>"

    def add(self, awaitable, task_id=None):
        if task_id is None:
            task_id = len(self.tasks.keys())
        task_id = str(task_id)
        if task_id in self.tasks.keys():
            raise Exception(f"Task {task_id} already exists")
        self.tasks[task_id] = {"awaitable": awaitable, "status": "pending"}
        return self.tasks[task_id]

    def __add__(self, awaitable, task_id=None):
        return self.add(awaitable, task_id=task_id)



    def info(self):
        print(repr(self)[:-1], end="\n")
        for k, task in self.tasks.items():
            print(f" - Task {k} ({task['status']}): {task['awaitable'].__name__}", end="")
            if "return" in task.keys():
                print(f" --> ({type(task['return']).__name__})", end="")
                if type(task["return"]) in [list, dict, tuple]:
                    print(" of length:", len(task["return"]), end="")
                else:
                    print(":",task["return"], end="")
            print("")
        print(">")


    async def run(self, raise_errors=True, return_dict=False):
        keys = []
        tasks = []
        for k, v in self.tasks.items():
            if v["status"] != "pending":
                continue
            keys.append(k)
            tasks.append(v["awaitable"])
            self.tasks[k]["status"] = "running"
        n_tasks = len(tasks)
        print(f" * AsyncPool: Running {n_tasks} tasks")
        ret = await asyncio.gather(*tasks, return_exceptions=True)
        errors = 0
        ok = 0
        for k, rv in zip(keys, ret):
            self.tasks[k]["return"] = rv
            if isinstance(rv, Exception):
                self.tasks[k]["status"] = "error"
                errors += 1
                bioiain.log("error", f"in task: {k}: {rv}")
                if raise_errors:
                    raise rv
            else:
                self.tasks[k]["status"] = "done"
                ok += 1
        print(f" * AsyncPool: Finished {ok+errors} tasks ({errors} errors)")
        if return_dict:
            return self.tasks
        return ret

    def without_errors(self, raise_errors=False, **kwargs):
        self.start(raise_errors=raise_errors, **kwargs)


    def start(self, raise_errors=True, **kwargs):
        asyncio.run(self.run(raise_errors=raise_errors, **kwargs))

    def get_return(self, key):
        return self.tasks[key]["return"]




pool = AsyncPool()
print(pool)

pool + simple(1, sleep=True)
pool + simple(2)
pool + test()
pool.add(simple("aaaa"), task_id="task1")
pool.add(simple("bbbb"), task_id="10")
print(pool)
print(pool.start(raise_errors=False))
pool.info()


