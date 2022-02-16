from collections import OrderedDict
from typing import Union


def is_dict(o):
    return isinstance(o, (dict, AttributeDict))


def recursive_dict_update(destination, origin, append: Union[str, bool] = False):
    for k, v in origin.items():
        dest_v = destination.get(k, None)
        if is_dict(v) and is_dict(dest_v):
            recursive_dict_update(destination[k], v)
        elif append and isinstance(v, list) and isinstance(dest_v, list):
            for list_v in v:
                append_needed = True
                if is_dict(list_v) and isinstance(append, str) and append in list_v:
                    key = list_v[append]
                    for i in range(len(dest_v)):
                        if is_dict(dest_v[i]) and dest_v[i] and dest_v[i].get(append, None) == key:
                            recursive_dict_update(dest_v[i], list_v, append=append)
                            append_needed = False
                if append_needed:
                    dest_v.append(list_v)
        else:
            destination[k] = v


def recursive_dict_map(dictionnary, function):
    r = {}
    for n, v in dictionnary.items():
        if is_dict(v):
            v = recursive_dict_map(v, function=function)
        else:
            v = function(n, v)
        if v is not None:
            r[n] = v
    return r


class AttributeDict(OrderedDict):
    @staticmethod
    def from_dict(d, recursive=False):
        if isinstance(d, AttributeDict):
            return d
        if isinstance(d, (OrderedDict, dict)):
            r = AttributeDict()
            for k, v in d.items():
                if is_dict(v) and recursive:
                    v = AttributeDict.from_dict(v, True)
                r[k] = v
            return r
        return d

    @staticmethod
    def from_json(json):
        from json import loads
        d = loads(json)
        return AttributeDict.from_dict(d, recursive=True)

    @staticmethod
    def from_yaml(yaml_doc):
        import yaml
        d = yaml.load(yaml_doc, Loader=yaml.FullLoader)
        return AttributeDict.from_dict(d, recursive=True)

    def to_dict(self):
        return recursive_dict_map(self, lambda k, v: v)

    def to_json(self):
        from json import dumps
        return dumps(self)

    def to_yaml(self, file=None):
        import yaml
        return yaml.dump(self.to_dict(), stream=file, default_flow_style=False)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise ValueError('Invalid AttributeDict key: %s.' % repr(key))
        if '.' in key:
            keys = key.split('.')
            r = self
            for i, k in enumerate(keys[:-1]):
                try:
                    r = r[k]
                except (KeyError, IndexError, TypeError):
                    raise IndexError(f'Invalid key: {".".join(keys[:i])}.') from None
            try:
                r[keys[-1]] = value
            except (KeyError, IndexError, TypeError):
                raise IndexError(f'Invalid key: {keys}.') from None
        super(AttributeDict, self).__setitem__(key, value)

    def __getitem__(self, item):
        if isinstance(item, int):
            k = list(self.keys())
            if item > len(k):
                raise IndexError('Index %i out of range (AttributeDict length: %s)' % (item, len(k)))
            return super(AttributeDict, self).__getitem__(list(self.keys())[item])
        elif isinstance(item, str):
            if '.' not in item:
                return super(AttributeDict, self).__getitem__(item)
            else:
                item = item.split('.')
                r = self
                for i, it in enumerate(item):
                    try:
                        r = r[it]
                    except (KeyError, IndexError, TypeError):
                        raise IndexError(f'Invalid item: {".".join(item[:i])}.') from None
                return r
        else:
            return super(AttributeDict, self).__getitem__(str(item))
        
    def __delitem__(self, item):
        if isinstance(item, int):
            k = list(self.keys())
            if item > len(k):
                raise IndexError('Index %i out of range (AttributeDict length: %s)' % (item, len(k)))
            return super(AttributeDict, self).__delitem__(list(self.keys())[item])
        elif isinstance(item, str):
            if '.' not in item:
                return super(AttributeDict, self).__delitem__(item)
            else:
                item = item.split('.')
                r = self
                for i, it in enumerate(item[:-1]):
                    try:
                        r = r[it]
                    except (KeyError, IndexError, TypeError):
                        raise IndexError(f'Invalid item: {".".join(item[:i])}.') from None
                del r[item[-1]]
        else:
            return super(AttributeDict, self).__delitem__(str(item))

    def __getattr__(self, item):
        if item in self:
            return self[item]
        raise AttributeError('%s is unknown' % item)

    def __setattr__(self, key, value):
        if key in self:
            self[key] = value
        raise AttributeError('%s is unknown' % key)

    def __iter__(self):
        for v in self.values():
            yield v

    def __len__(self):
        return len(self.keys())

    def recursive_update(self, d):
        recursive_dict_update(self, d)
        return self

    def filter(self, condition, recursive=False):
        for k in list(self.keys()):
            v = self[k]
            if recursive and isinstance(v, AttributeDict):
                v.filter(condition, True)
            elif not condition(k, v):
                del self[k]
        return self

    def map(self, f, recursive=False, remove_if_none=False):
        for k, v in self.items():
            if recursive and isinstance(v, AttributeDict):
                v.map(f, True)
            else:
                r = f(k, v)
                if remove_if_none and r is None:
                    del self[k]
                else:
                    self[k] = r
        return self

    def walk(self):
        for k, v in self.items():
            if isinstance(v, AttributeDict):
                for s in v.walk():
                    yield f"{k}.{s}"
            else:
                yield k

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def subset(self, items):
        from copy import deepcopy
        r = AttributeDict()
        if isinstance(items, str):
            items = items.split(',')
            items = [_.strip() for _ in items]
        for it in items:
            r[it] = deepcopy(self[it])
        return r

    def check(self, path, value=True, missing=None):
        path = path.split('.')

        miss = False
        item = self
        for i, p in enumerate(path):
            if p not in item:
                miss = True
                path = '.'.join(path[:i])
                break
            item = item[p]

        if miss:
            if missing is None:
                raise AttributeError(f'Missing attribute {path}.')
            return missing

        if isinstance(value, bool) or value is None:
            return item is value
        if isinstance(value, tuple) and not isinstance(item, tuple):
            return item in value
        return item == value

    def get(self, path, default='raise'):
        if isinstance(path, str) and ',' in path:
            path = set(path.split(','))
        if isinstance(path, set):
            r = {}
            for p in path:
                try:
                    r[p] = self.get(p, default)
                except AttributeError:
                    continue
            return AttributeDict.from_dict(r)
        elif isinstance(path, (list, tuple)):
            return [self.get(p, default) for p in path]
        
        path = path.strip().split('.')

        miss = False
        item = self
        for i, p in enumerate(path):
            try:
                item = item[p]
            except (KeyError, IndexError, TypeError):
                miss = True
                path = '.'.join(path[:i])
                break

        if miss:
            if default is 'raise':
                raise AttributeError(f'Missing attribute {path}.')
            return default

        return item

    def pop(self, path):
        r = self.get(path)
        if isinstance(path, set) or (isinstance(path, str) and ',' in path):
            for key in r.keys():
                del self[key]
        elif isinstance(path, (list, tuple)):
            for key in path:
                del self[key]
        else:
            del self[path]
        return r