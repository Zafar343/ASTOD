import json
import os


_path = '/home/zafar/old_pc/data_sets/coco2017/annotations/semi_supervised/instances_train2017.1@10-unlabeled.json'
with open(_path, "r") as f:
    data   =  json.load(f)
    _data  =  dict.fromkeys(data)

    _data["images"]      =  data["images"][:10]
    _data["annotations"] =  data["annotations"][:10]
    _data["licenses"]   =  data["licenses"]
    _data["categories"]  =  data["categories"]
    _data["info"]        =  data["info"]

_path = _path.split('-unlabeled')[0]
_path = _path + '-unlabeled_new.json'

with open(_path, 'w') as f:
    json.dump(_data, f)    
print('done')