import numpy as np
import plyfile

def save_ply(xyz, features_dc, features_rest, opacities, scale, rotation, path):
    

    def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']        # All channels except the 3 DC
        for i in range(features_dc.shape[1]*features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(features_rest.shape[1]*features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    import os; os.makedirs(os.path.dirname(path), exist_ok=True)
    normals_zero = np.zeros_like(xyz.numpy())
    features_dc_flat = features_dc.transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
    features_rest_flat = features_rest.transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
    attributes = np.concatenate((xyz.numpy(), normals_zero, features_dc_flat, features_rest_flat, opacities.numpy(), scale.numpy(), rotation.numpy()), axis=1)
    elements = np.empty(xyz.shape[0], dtype=[(attribute, 'f4') for attribute in construct_list_of_attributes()])
    elements[:] = list(map(tuple, attributes))
    el = plyfile.PlyElement.describe(elements, 'vertex')
    plyfile.PlyData([el]).write(path)

