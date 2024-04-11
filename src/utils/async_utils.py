import asyncio


async def gather_with_concurrency(n, *coros, task_timeout=200, gather_timeout=3600):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            try:
                # Here we apply a timeout to each individual task
                return await asyncio.wait_for(coro, timeout=task_timeout)
            except asyncio.TimeoutError:
                print(f"A task has timed out after {task_timeout} seconds.")
                # Handle the individual task timeout (e.g., by returning a sentinel value)
                return None

    try:
        # Here we apply a timeout to the entire gather operation
        return await asyncio.wait_for(asyncio.gather(*(sem_coro(c) for c in coros)), timeout=gather_timeout)
    except asyncio.TimeoutError:
        print(f"The gather operation timed out after {gather_timeout} seconds!")
        # Handle the gather timeout (e.g., by returning a sentinel value or performing cleanup)
        return []