import json
import uuid

import numpy as np
#data_dict["joint_type"]+
def write_data(root, data_dict,i):
    scene_id = uuid.uuid4().hex
    path = root / "scenes" / ("_"+str(i)+"_"+scene_id +"_index_"+str(data_dict["joint_index"])+ "_.npz")
    assert not path.exists()
    np.savez_compressed(path, **data_dict)
    return scene_id