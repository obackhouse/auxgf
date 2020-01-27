''' Decorators for class methods.
'''


def record_time(key):
    def decorator(method):
        def wrapper(cls):
            cls._timer()
            out = method(cls)
            cls._timings[key] = cls._timings.get(key, 0.0) + cls._timer()
            return out
        return wrapper
    return decorator


def record_energy(key):
    def decorator(method):
        def wrapper(cls):
            e = method(cls)
            cls._energies[key] = cls._energies.get(key, []) + [e,]
            return e
        return wrapper
    return decorator
