class Arg:

    _args: dict = None
    def __new__(cls):
        if cls._args is None:
            cls._args = cls.get_default_conf()
        return cls

    @classmethod
    def get(cls, key, default = None):
        if key not in cls._args:
            if default is not None:
                return default
            raise Exception(f"Please set the {key} argument")
        return cls._args[key]

    @classmethod
    def set(cls, key, value):
        if key not in cls._args:
            raise Exception(f"Please set the {key} argument")
        cls._args[key] = value