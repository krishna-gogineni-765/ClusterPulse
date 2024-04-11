import numbers
import os
from functools import lru_cache

import yaml
from cryptography.fernet import Fernet

from src.utils.metaclasses import Singleton


class Config:
    def __init__(self, config_dict):
        if config_dict is None:
            raise TypeError("Config is null")
        self.__source = config_dict

    def properties(self):
        return self.__source

    def property_value(self, property_name):
        if property_name in self.__source:
            return self.__source[property_name]
        else:
            raise KeyError(f'property {property_name} does not exist in config')

    def update_value(self, property_name, new_value):
        if property_name in self.__source:
            self.__source[property_name] = new_value
        else:
            raise KeyError(f'property {property_name} does not exist in config')


class ConfigClient(metaclass=Singleton):

    def __init__(self, secret_key):
        self.__secret_key = secret_key
        self.cryptor = Fernet(secret_key)

    @staticmethod
    def __convert_yaml_to_props(d, sep="."):
        obj = dict()
        try:
            def recurse(t, parent_key=""):
                if isinstance(t, list):
                    for i in range(len(t)):
                        recurse(t[i], parent_key + f'[{i}]')
                elif isinstance(t, dict):
                    for k, v in t.items():
                        append_key = sep + str(k) if parent_key else str(k)
                        if isinstance(k, numbers.Number):
                            append_key = f'[{k}]'
                        recurse(v, parent_key + append_key)
                else:
                    obj[parent_key] = t if (t is not None) else ''

            recurse(d)
        except:
            raise Exception("Failed to convert yaml-config to props-config format")
        return obj

    def get_config_from_local_file(self, config_file_path):
        try:
            with open(config_file_path, 'r') as stream:
                yaml_config = yaml.safe_load(stream)
        except:
            raise Exception("Failed to load/parse config file")
        props_config = ConfigClient.__convert_yaml_to_props(yaml_config)
        decrypted_config = self.__decrypt_config(props_config)
        return Config(decrypted_config)

    def __decrypt_config(self, source):
        for key, value in source.items():
            if isinstance(value, str):
                if value.startswith('enc'):
                    source[key] = self.__get_decrypted_text(key, value[4:-1]).decode()
        return source

    def __get_decrypted_text(self, identifier_key, encrypted_text):
        try:
            return self.cryptor.decrypt(encrypted_text.encode())
        except:
            raise Exception(f"Config decryption failed for key: {identifier_key}!")


class ConfigManager(metaclass=Singleton):

    def __init__(self, host=None, secret_key=None):
        self.__config_store = {}
        self.config_client = ConfigClient(secret_key)

    def __update_local_config(self, service_name, config_file_path):
        key = self.__generate_local_key(service_name)
        self.__config_store[key] = ConfigClient().get_config_from_local_file(config_file_path)

    def get_local_config(self, service_name, config_file_path=None):
        key = self.__generate_local_key(service_name)
        if key not in self.__config_store:
            self.force_update_local_config(service_name, config_file_path)
        return self.__config_store[key]

    def force_update_local_config(self, service_name, config_file_path):
        if not config_file_path:
            raise TypeError("config_file_path is null/empty")
        self.__update_local_config(service_name, config_file_path)

    @staticmethod
    def __generate_local_key(service_name):
        if not service_name:
            raise TypeError("service_name is null/empty")
        return f"{service_name}/local"


@lru_cache(maxsize=1)
def load_config_object():
    if 'DECRYPTION_SECRET_KEY' not in os.environ:
        raise ValueError("Please start the docker with the decryption secret key as environment variable")
    config_manager = ConfigManager(None, os.environ['DECRYPTION_SECRET_KEY'])
    config_object = config_manager.get_local_config('conversation-maps-config',
                                                    config_file_path="/Users/krishna/Desktop/Projects/conversational-maps/from-scratch/config/config.yml")
    return config_object


def encrypt_value(value):
    return Fernet(os.environ["DECRYPTION_SECRET_KEY"]).encrypt(value.encode()).decode()

# os.environ["DECRYPTION_SECRET_KEY"] = "secret"
config_properties = load_config_object().properties()

