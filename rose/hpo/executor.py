from concurrent.futures import ProcessPoolExecutor, as_completed

class LocalParallelExecutor: 
    def __init__(self, max_workers: int):
        self.pool = ProcessPoolExecutor(max_workers=max_workers)
    
    def run_batch(self, fn, params_list):
        futures = {self.pool.submit(fn, params): params for params in params_list}
        results = []
        for future in as_completed(futures):
            params = futures[future]
            metric = future.result()
            results.append({"params": params, 
                            "metric": metric})
        return results