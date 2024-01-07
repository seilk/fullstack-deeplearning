def accThroughOnlyOnce(rank, func):
    from main import accelerator
    def wrapper(*args, **kwargs):
        if accelerator.is_main_process:
            func(*args, **kwargs)
    return wrapper