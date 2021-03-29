import numpy as np
from InterPhon.core import UnitCell

# Searching lattice point group operations
W_candidate = [np.array([[0, 1, 0],
                         [1, 0, 0],
                         [0, 0, 1]]), np.array([[0, 1, 0],
                                                [-1, 0, 0],
                                                [0, 0, 1]]), np.array([[0, -1, 0],
                                                                       [1, 0, 0],
                                                                       [0, 0, 1]]), np.array([[0, -1, 0],
                                                                                              [-1, 0, 0],
                                                                                              [0, 0, 1]]),
               np.array([[0, 1, 0],
                         [1, 1, 0],
                         [0, 0, 1]]), np.array([[0, 1, 0],
                                                [1, -1, 0],
                                                [0, 0, 1]]), np.array([[0, 1, 0],
                                                                       [-1, 1, 0],
                                                                       [0, 0, 1]]), np.array([[0, -1, 0],
                                                                                              [1, 1, 0],
                                                                                              [0, 0, 1]]), np.array([[0, -1, 0],
                                                                                                                     [-1, 1, 0],
                                                                                                                     [0, 0, 1]]), np.array([[0, -1, 0],
                                                                                                                                            [1, -1, 0],
                                                                                                                                            [0, 0, 1]]), np.array([[0, 1, 0],
                                                                                                                                                                   [-1, -1, 0],
                                                                                                                                                                   [0, 0, 1]]), np.array([[0, -1, 0],
                                                                                                                                                                                          [-1, -1, 0],
                                                                                                                                                                                          [0, 0, 1]]),
               np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]]), np.array([[1, 0, 0],
                                                [0, -1, 0],
                                                [0, 0, 1]]), np.array([[-1, 0, 0],
                                                                       [0, 1, 0],
                                                                       [0, 0, 1]]), np.array([[-1, 0, 0],
                                                                                              [0, -1, 0],
                                                                                              [0, 0, 1]]),
               np.array([[1, 0, 0],
                         [1, 1, 0],
                         [0, 0, 1]]), np.array([[1, 0, 0],
                                                [1, -1, 0],
                                                [0, 0, 1]]), np.array([[1, 0, 0],
                                                                       [-1, 1, 0],
                                                                       [0, 0, 1]]), np.array([[-1, 0, 0],
                                                                                              [1, 1, 0],
                                                                                              [0, 0, 1]]), np.array([[-1, 0, 0],
                                                                                                                     [-1, 1, 0],
                                                                                                                     [0, 0, 1]]), np.array([[-1, 0, 0],
                                                                                                                                            [1, -1, 0],
                                                                                                                                            [0, 0, 1]]), np.array([[1, 0, 0],
                                                                                                                                                                   [-1, -1, 0],
                                                                                                                                                                   [0, 0, 1]]), np.array([[-1, 0, 0],
                                                                                                                                                                                          [-1, -1, 0],
                                                                                                                                                                                          [0, 0, 1]]),
               np.array([[1, 1, 0],
                         [0, 1, 0],
                         [0, 0, 1]]), np.array([[1, 1, 0],
                                                [0, -1, 0],
                                                [0, 0, 1]]), np.array([[1, -1, 0],
                                                                       [0, 1, 0],
                                                                       [0, 0, 1]]), np.array([[-1, 1, 0],
                                                                                              [0, 1, 0],
                                                                                              [0, 0, 1]]), np.array([[-1, -1, 0],
                                                                                                                     [0, 1, 0],
                                                                                                                     [0, 0, 1]]), np.array([[-1, 1, 0],
                                                                                                                                            [0, -1, 0],
                                                                                                                                            [0, 0, 1]]), np.array([[1, -1, 0],
                                                                                                                                                                   [0, -1, 0],
                                                                                                                                                                   [0, 0, 1]]), np.array([[-1, -1, 0],
                                                                                                                                                                                          [0, -1, 0],
                                                                                                                                                                                          [0, 0, 1]]),
               np.array([[1, 1, 0],
                         [1, 0, 0],
                         [0, 0, 1]]), np.array([[1, 1, 0],
                                                [-1, 0, 0],
                                                [0, 0, 1]]), np.array([[1, -1, 0],
                                                                       [1, 0, 0],
                                                                       [0, 0, 1]]), np.array([[-1, 1, 0],
                                                                                              [1, 0, 0],
                                                                                              [0, 0, 1]]), np.array([[-1, -1, 0],
                                                                                                                     [1, 0, 0],
                                                                                                                     [0, 0, 1]]), np.array([[-1, 1, 0],
                                                                                                                                            [-1, 0, 0],
                                                                                                                                            [0, 0, 1]]), np.array([[1, -1, 0],
                                                                                                                                                                   [-1, 0, 0],
                                                                                                                                                                   [0, 0, 1]]), np.array([[-1, -1, 0],
                                                                                                                                                                                          [-1, 0, 0],
                                                                                                                                                                                          [0, 0, 1]]),
               ]


def symmetry_2d(unit_cell=UnitCell()):
    print(unit_cell.atom_type)
    print(unit_cell.atom_true)
    atom_true_original = np.transpose(unit_cell.atom_direct[unit_cell.atom_true, :])

    # metric tensor
    G_metric = np.dot(unit_cell.lattice_matrix[0:2, 0:2], np.transpose(unit_cell.lattice_matrix[0:2, 0:2]))

    rot_ind = []
    for ind, rot in enumerate(W_candidate):
        G_rotate = np.dot(np.transpose(rot[0:2, 0:2]), np.dot(G_metric, rot[0:2, 0:2]))
        if np.allclose(G_metric, G_rotate, atol=1e-06):
            rot_ind.append(ind)
    # print(rot_ind)

    w_for_given_rot = []
    same_index = []
    for ind in rot_ind:
        atom_true_rot = np.dot(W_candidate[ind], atom_true_original)

        # space group search
        w_candidate = [np.array([0.0, 0.0, 0.0])]
        # for index, value in enumerate(unit_cell.atom_true):
        #     other_atom_ind = [i for i in unit_cell.atom_true]
        #     other_atom_ind.remove(value)
        #     for other_index, other_ind in enumerate(other_atom_ind):
        #         if unit_cell.atom_type[other_ind] == unit_cell.atom_type[value]:
        #             w = np.round_(atom_true_original[:, other_index] - atom_true_rot[:, index], 6)
        #             if w[2] == 0.0:
        #                 w_, _ = np.modf(w)
        #                 if not np.allclose(w_, np.zeros([3, ]), atol=1e-06):
        #                     w_candidate.append(w_)
        # print('w_candidate=', w_candidate)
        # print(np.array(w_candidate).shape)

        trans_for_given_rot = []
        _same_index = []
        for w in w_candidate:
            atom_transform = atom_true_rot + w.reshape([3, 1])
            __same_index = []
            for index, value in enumerate(unit_cell.atom_true):
                same_atom_type = [ind_ for ind_, val_ in enumerate(unit_cell.atom_true)
                                  if unit_cell.atom_type[val_] == unit_cell.atom_type[value]]

                for _, same_atom_index in enumerate(same_atom_type):
                    delta_x = atom_transform[:, index] - atom_true_original[:, same_atom_index]  # atom-to-atom comparison
                    delta_x_cart = np.matmul(np.transpose(unit_cell.lattice_matrix), delta_x - np.rint(delta_x))

                    if np.allclose(delta_x_cart, np.zeros([3, ]), atol=1e-06):
                        if same_atom_index not in __same_index:
                            __same_index.append(same_atom_index)
                            break

            if len(__same_index) == len(unit_cell.atom_true):
                trans_for_given_rot.append(w)
                _same_index.append(__same_index)

        w_for_given_rot.append(trans_for_given_rot)
        same_index.append(_same_index)

    W_select = []
    w_select = []
    same_index_select = []
    look_up_table = np.array([0, 0, 0, 0, 0, 0])  # num of following point-group operations: m, 1, 2, 3, 4, 6
    for ind_ind, _rot_ind in enumerate(rot_ind):
        if w_for_given_rot[ind_ind]:
            W_select.append(W_candidate[_rot_ind])
            w_select.append(w_for_given_rot[ind_ind])
            same_index_select.append(same_index[ind_ind])

            look_up = (np.trace(W_candidate[_rot_ind][0:2, 0:2]), np.linalg.det(W_candidate[_rot_ind][0:2, 0:2]))
            if look_up == (0.0, -1.0):
                look_up_table[0] += 1
            elif look_up == (2.0, 1.0):
                look_up_table[1] += 1
            elif look_up == (-2.0, 1.0):
                look_up_table[2] += 1
            elif look_up == (-1.0, 1.0):
                look_up_table[3] += 1
            elif look_up == (0.0, 1.0):
                look_up_table[4] += 1
            elif look_up == (1.0, 1.0):
                look_up_table[5] += 1
            else:
                print('What is this operation?')
                assert False

    if np.allclose(look_up_table, np.array([0, 1, 0, 0, 0, 0])):
        print('Point group = 1')
    elif np.allclose(look_up_table, np.array([0, 1, 1, 0, 0, 0])):
        print('Point group = 2')
    elif np.allclose(look_up_table, np.array([1, 1, 0, 0, 0, 0])):
        print('Point group = m')
    elif np.allclose(look_up_table, np.array([2, 1, 1, 0, 0, 0])):
        print('Point group = 2mm')
    elif np.allclose(look_up_table, np.array([2, 2, 0, 0, 0, 0])):
        print('Point group = m (cm)')
    elif np.allclose(look_up_table, np.array([4, 2, 2, 0, 0, 0])):
        print('Point group = 2mm (c2mm)')
    elif np.allclose(look_up_table, np.array([0, 1, 1, 0, 2, 0])):
        print('Point group = 4')
    elif np.allclose(look_up_table, np.array([4, 1, 1, 0, 2, 0])):
        print('Point group = 4mm')
    elif np.allclose(look_up_table, np.array([0, 1, 0, 2, 0, 0])):
        print('Point group = 3')
    elif np.allclose(look_up_table, np.array([3, 1, 0, 2, 0, 0])):
        print('Point group = 3m')
    elif np.allclose(look_up_table, np.array([0, 1, 1, 2, 0, 2])):
        print('Point group = 6')
    elif np.allclose(look_up_table, np.array([6, 1, 1, 2, 0, 2])):
        print('Point group = 6mm')
    else:
        print('What is this point group?')
        assert False

    print('W_select:', W_select)
    print('w_select:', w_select)
    print('same_index_select:', same_index_select)

    require = []
    not_require = []
    point_group_ind = []

    for _ind, _ in enumerate(unit_cell.atom_true):
        if require:
            found_flag = False
            for W_ind, same in enumerate(same_index_select):
                if same[0][_ind] in require:
                    point_group_ind.append(W_ind)
                    not_require.append(_ind)
                    found_flag = True
                    break
            if not found_flag:
                require.append(_ind)
        else:
            require.append(_ind)

    # for W_ind, _ in enumerate(W_select):
    #     for _, _same in enumerate(same_index_select[W_ind]):
    #         for __ind, __same in enumerate(_same):
    #             if __ind < __same:
    #                 if __ind not in not_require:
    #                     not_require.append(__ind)
    #                     point_group_ind.append(W_ind)

    # not_require = np.array(not_require)
    # point_group_ind = np.array(point_group_ind)
    # _sorted_indices = np.argsort(not_require)
    #
    # not_require = not_require[_sorted_indices]
    # point_group_ind = point_group_ind[_sorted_indices]

    return W_select, w_select, same_index_select, point_group_ind, require, not_require
