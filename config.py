import json 

class Config:

    _conf: dict = None
    def __new__(cls):
        if Config._conf is None:
            Config._conf = Config.get_default_conf()
        return Config

    @classmethod
    def update_conf(cls, conf):
        cls._conf = conf

    @classmethod
    def get(cls, key, default = None):
        if key not in cls._conf:
            if default is not None:
                return default
            raise Exception(f"Please set the {key} variable in the config file")
        return cls._conf[key]

    @classmethod
    def __getitem__(cls, key):
        return Config.get_conf(key)

    @classmethod
    def set(cls, key, value):
        if key not in cls._conf:
            raise Exception(f"Please set the {key} variable in the config file")
        cls._conf[key] = value

    @classmethod
    def __setitem__(cls, key, value):
        Config.set_conf(key, value)


    @classmethod
    def from_file(cls, filepath):
        cls._conf = json.load(open(filepath))
        cls._conf["CONF_PATH"] = filepath

    @classmethod
    def get_default_conf(cls):
        return {
            "GEN_DATA" : 0,
            "REAL_BEHAVIORAL_DATA" : 1,
            "CACHED_BEHAVIORAL_DATA" : 1,
            "SHOW_PLOTS" : 0,
            "PER_BAT_ANALYSIS" : 0,
            "FE" : 1,
            "WITH_PAIRS" : 1, 
            "BINNING" : 0,
            "NOISE_CANCELLATION" : 0,
            "SQRT_WEIGHT" : 0,
            "OVERWRITE" : 0,
            
            "CV" : 3,
            "N_SHUFFLES" : 5,
            "FI_NUM" : 7,
            "UPSAMPLING": 1,
            "POPUP_RESULTS" : 1,
            "MIN_SPIKES" : 500,
            "MIN_DATAPOINTS" : 50000,
            "STORE_DATAFRAME": 0,
            "STORE_NORMALIZED_DATAFRAME": 0,
            "CONF_PATH" : -1
        }