# ldpc.py  —— Windows/Linux 兼容的 ctypes 加强版
import os
import ctypes as ct
import numpy as np


class code:
    def __init__(self, standard='802.11n', rate='1/2', z=27, ptype='A'):
        self.standard = standard
        self.rate = rate
        self.z = z
        self.ptype = ptype
        self.proto = self.assign_proto()
        vdeg, cdeg, intrlv = self.prepare_decoder()
        # ensure we store as numpy arrays for other python-side code,
        # but we'll convert to ctypes arrays when calling C
        self.vdeg = np.array(vdeg, dtype=np.int64)
        self.cdeg = np.array(cdeg, dtype=np.int64)
        self.intrlv = np.array(intrlv, dtype=np.int64)
        self.Nv = len(vdeg)
        self.Nc = len(cdeg)
        self.Nmsg = len(intrlv)
        self.N = self.Nv
        self.K = self.Nv - self.Nc

        # load C library (ctypes) once per instance
        self._load_clib()
        # 在 __init__ 的结尾处加入
        self.print_iter = getattr(self, 'print_iter', False)  # 置 True 时，decode 每次调用都会打印内部迭代次数

        # DGL/CUDA 相关可调参数（可在外部覆盖）
        self.dgl_device = getattr(self, 'dgl_device', 'cuda')
        self.dgl_llr_clip = float(getattr(self, 'dgl_llr_clip', 20.0))
        self.dgl_max_iter = int(getattr(self, 'dgl_max_iter', 50))
        self.dgl_verbose = bool(getattr(self, 'dgl_verbose', False))
        self.dgl_log_every = int(getattr(self, 'dgl_log_every', 1))
        self.dgl_check_every = int(getattr(self, 'dgl_check_every', 1))
        self.dgl_dtype = getattr(self, 'dgl_dtype', 'float32')  # 'float32' | 'float16' | 'bfloat16'
        self.dgl_microbatch = getattr(self, 'dgl_microbatch', None)  # int 或 None；避免一次性 batch 太大
        self.dgl_stop_all = getattr(self, 'dgl_stop_all', True)  # True：全部校验通过才早停；False：统一跑满迭代
        return

    def assign_proto(self):
        """ Generates arrays to enable the construction of
        IEEE standard-compliant LDPC codes

        Parameters
        ----------
        standard: string
        Specifies the IEEE standard used, 802.11n or 802.16

        rate: string
        Specifies the code rate, 1/2, 2/3, 3/4 or 5/6

        z: int
        Optional parameter (not needed for for 802.16, required for 802.11n)
        Specifies the protograph expansion factor, freely chooseable >= 3 for
        IEEE 802.16, restricted to (27, 54, 81) for IEEE 802.11n

        ptype: character
        Optional parameter.
        Either A or B for 802.16 rates 2/3 and 3/4 where two options are
        specified in the standard. Parameter unused for all other codes.

        Returns
        -------
        np.ndarray
        Protograph for an LDPC parity-check matrix
        """

        standard = self.standard
        rate = self.rate
        z = self.z
        ptype = self.ptype

        if standard == "802.16":
            # N = z*24
            if rate == '1/2':
                proto = np.array([
                    [-1, 94, 73, -1, -1, -1, -1, -1, 55, 83, -1, -1, 7, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, 27, -1, -1, -1, 22, 79, 9, -1, -1, -1, 12, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, 24, 22, 81, -1, 33, -1, -1, -1, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
                    [61, -1, 47, -1, -1, -1, -1, -1, 65, 25, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, 39, -1, -1, -1, 84, -1, -1, 41, 72, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, 46, 40, -1, 82, -1, -1, -1, 79, 0, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
                    [-1, -1, 95, 53, -1, -1, -1, -1, -1, 14, 18, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
                    [-1, 11, 73, -1, -1, -1, 2, -1, -1, 47, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
                    [12, -1, -1, -1, 83, 24, -1, 43, -1, -1, -1, 51, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1],
                    [-1, -1, -1, -1, -1, 94, -1, 59, -1, -1, 70, 72, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
                    [-1, -1, 7, 65, -1, -1, -1, -1, 39, 49, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
                    [43, -1, -1, -1, -1, 66, -1, 41, -1, -1, -1, 26, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]
                ])
            elif rate == '2/3':
                if ptype == 'A':
                    proto = np.array([
                        [3, 0, -1, -1, 2, 0, -1, 3, 7, -1, 1, 1, -1, -1, -1, -1, 1, 0, -1, -1, -1, -1, -1, -1],
                        [-1, -1, 1, -1, 36, -1, -1, 34, 10, -1, -1, 18, 2, -1, 3, 0, -1, 0, 0, -1, -1, -1, -1, -1],
                        [-1, -1, 12, 2, -1, 15, -1, 40, -1, 3, -1, 15, -1, 2, 13, -1, -1, -1, 0, 0, -1, -1, -1, -1],
                        [-1, -1, 19, 24, -1, 3, 0, -1, 6, -1, 17, -1, -1, -1, 8, 39, -1, -1, -1, 0, 0, -1, -1, -1],
                        [20, -1, 6, -1, -1, 10, 29, -1, -1, 28, -1, 14, -1, 38, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1],
                        [-1, -1, 10, -1, 28, 20, -1, -1, 8, -1, 36, -1, 9, -1, 21, 45, -1, -1, -1, -1, -1, 0, 0, -1],
                        [35, 25, -1, 37, -1, 21, -1, -1, 5, -1, -1, 0, -1, 4, 20, -1, -1, -1, -1, -1, -1, -1, 0, 0],
                        [-1, 6, 6, -1, -1, -1, 4, -1, 14, 30, -1, 3, 36, -1, 14, -1, 1, -1, -1, -1, -1, -1, -1, 0]
                    ])
                elif ptype == 'B':
                    proto = np.array([
                        [2, -1, 19, -1, 47, -1, 48, -1, 36, -1, 82, -1, 47, -1, 15, -1, 95, 0, -1, -1, -1, -1, -1, -1],
                        [-1, 69, -1, 88, -1, 33, -1, 3, -1, 16, -1, 37, -1, 40, -1, 48, -1, 0, 0, -1, -1, -1, -1, -1],
                        [10, -1, 86, -1, 62, -1, 28, -1, 85, -1, 16, -1, 34, -1, 73, -1, -1, -1, 0, 0, -1, -1, -1, -1],
                        [-1, 28, -1, 32, -1, 81, -1, 27, -1, 88, -1, 5, -1, 56, -1, 37, -1, -1, -1, 0, 0, -1, -1, -1],
                        [23, -1, 29, -1, 15, -1, 30, -1, 66, -1, 24, -1, 50, -1, 62, -1, -1, -1, -1, -1, 0, 0, -1, -1],
                        [-1, 30, -1, 65, -1, 54, -1, 14, -1, 0, -1, 30, -1, 74, -1, 0, -1, -1, -1, -1, -1, 0, 0, -1],
                        [32, -1, 0, -1, 15, -1, 56, -1, 85, -1, 5, -1, 6, -1, 52, -1, 0, -1, -1, -1, -1, -1, 0, 0],
                        [-1, 0, -1, 47, -1, 13, -1, 61, -1, 84, -1, 55, -1, 78, -1, 41, 95, -1, -1, -1, -1, -1, -1, 0]
                    ])
                else:
                    raise NameError('802.16 type must be either A or B')
            elif rate == '3/4':
                if ptype == 'A':
                    proto = np.array([
                        [6, 38, 3, 93, -1, -1, -1, 30, 70, -1, 86, -1, 37, 38, 4, 11, -1, 46, 48, 0, -1, -1, -1, -1],
                        [62, 94, 19, 84, -1, 92, 78, -1, 15, -1, -1, 92, -1, 45, 24, 32, 30, -1, -1, 0, 0, -1, -1, -1],
                        [71, -1, 55, -1, 12, 66, 45, 79, -1, 78, -1, -1, 10, -1, 22, 55, 70, 82, -1, -1, 0, 0, -1, -1],
                        [38, 61, -1, 66, 9, 73, 47, 64, -1, 39, 61, 43, -1, -1, -1, -1, 95, 32, 0, -1, -1, 0, 0, -1],
                        [-1, -1, -1, -1, 32, 52, 55, 80, 95, 22, 6, 51, 24, 90, 44, 20, -1, -1, -1, -1, -1, -1, 0, 0],
                        [-1, 63, 31, 88, 20, -1, -1, -1, 6, 40, 56, 16, 71, 53, -1, -1, 27, 26, 48, -1, -1, -1, -1, 0]
                    ])
                elif ptype == 'B':
                    proto = np.array([
                        [-1, 81, -1, 28, -1, -1, 14, 25, 17, -1, -1, 85, 29, 52, 78, 95, 22, 92, 0, 0, -1, -1, -1, -1],
                        [42, -1, 14, 68, 32, -1, -1, -1, -1, 70, 43, 11, 36, 40, 33, 57, 38, 24, -1, 0, 0, -1, -1, -1],
                        [-1, -1, 20, -1, -1, 63, 39, -1, 70, 67, -1, 38, 4, 72, 47, 29, 60, 5, 80, -1, 0, 0, -1, -1],
                        [64, 2, -1, -1, 63, -1, -1, 3, 51, -1, 81, 15, 94, 9, 85, 36, 14, 19, -1, -1, -1, 0, 0, -1],
                        [-1, 53, 60, 80, -1, 26, 75, -1, -1, -1, -1, 86, 77, 1, 3, 72, 60, 25, -1, -1, -1, -1, 0, 0],
                        [77, -1, -1, -1, 15, 28, -1, 35, -1, 72, 30, 68, 85, 84, 26, 64, 11, 89, 0, -1, -1, -1, -1, 0]
                    ])
                else:
                    raise NameError('802.16 type must be either A or B')
            elif rate == '5/6':
                proto = np.array([
                    [1, 25, 55, -1, 47, 4, -1, 91, 84, 8, 86, 52, 82, 33, 5, 0, 36, 20, 4, 77, 80, 0, -1, -1],
                    [-1, 6, -1, 36, 40, 47, 12, 79, 47, -1, 41, 21, 12, 71, 14, 72, 0, 44, 49, 0, 0, 0, 0, -1],
                    [51, 81, 83, 4, 67, -1, 21, -1, 31, 24, 91, 61, 81, 9, 86, 78, 60, 88, 67, 15, -1, -1, 0, 0],
                    [50, -1, 50, 15, -1, 36, 13, 10, 11, 20, 53, 90, 29, 92, 57, 30, 84, 92, 11, 66, 80, -1, -1, 0]
                ])
            else:
                raise NameError('802.16 invalid rate')
        elif standard == "802.11n":
            if z == 27:
                # N = 648
                if rate == '1/2':
                    proto = np.array([
                        [0, -1, -1, -1, 0, 0, -1, -1, 0, -1, -1, 0, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [22, 0, -1, -1, 17, -1, 0, 0, 12, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [6, -1, 0, -1, 10, -1, -1, -1, 24, -1, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
                        [2, -1, -1, 0, 20, -1, -1, -1, 25, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
                        [23, -1, -1, -1, 3, -1, -1, -1, 0, -1, 9, 11, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1],
                        [24, -1, 23, 1, 17, -1, 3, -1, 10, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
                        [25, -1, -1, -1, 8, -1, -1, -1, 7, 18, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
                        [13, 24, -1, -1, 0, -1, 8, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
                        [7, 20, -1, 16, 22, 10, -1, -1, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1],
                        [11, -1, -1, -1, 19, -1, -1, -1, 13, -1, 3, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
                        [25, -1, 8, -1, 23, 18, -1, 14, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
                        [3, -1, -1, -1, 16, -1, -1, 2, 25, 5, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]
                    ])
                elif rate == '2/3':
                    proto = np.array([
                        [25, 26, 14, -1, 20, -1, 2, -1, 4, -1, -1, 8, -1, 16, -1, 18, 1, 0, -1, -1, -1, -1, -1, -1],
                        [10, 9, 15, 11, -1, 0, -1, 1, -1, -1, 18, -1, 8, -1, 10, -1, -1, 0, 0, -1, -1, -1, -1, -1],
                        [16, 2, 20, 26, 21, -1, 6, -1, 1, 26, -1, 7, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
                        [10, 13, 5, 0, -1, 3, -1, 7, -1, -1, 26, -1, -1, 13, -1, 16, -1, -1, -1, 0, 0, -1, -1, -1],
                        [23, 14, 24, -1, 12, -1, 19, -1, 17, -1, -1, -1, 20, -1, 21, -1, 0, -1, -1, -1, 0, 0, -1, -1],
                        [6, 22, 9, 20, -1, 25, -1, 17, -1, 8, -1, 14, -1, 18, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
                        [14, 23, 21, 11, 20, -1, 24, -1, 18, -1, 19, -1, -1, -1, -1, 22, -1, -1, -1, -1, -1, -1, 0, 0],
                        [17, 11, 11, 20, -1, 21, -1, 26, -1, 3, -1, -1, 18, -1, 26, -1, 1, -1, -1, -1, -1, -1, -1, 0]
                    ])
                elif rate == '3/4':
                    proto = np.array([
                        [16, 17, 22, 24, 9, 3, 14, -1, 4, 2, 7, -1, 26, -1, 2, -1, 21, -1, 1, 0, -1, -1, -1, -1],
                        [25, 12, 12, 3, 3, 26, 6, 21, -1, 15, 22, -1, 15, -1, 4, -1, -1, 16, -1, 0, 0, -1, -1, -1],
                        [25, 18, 26, 16, 22, 23, 9, -1, 0, -1, 4, -1, 4, -1, 8, 23, 11, -1, -1, -1, 0, 0, -1, -1],
                        [9, 7, 0, 1, 17, -1, -1, 7, 3, -1, 3, 23, -1, 16, -1, -1, 21, -1, 0, -1, -1, 0, 0, -1],
                        [24, 5, 26, 7, 1, -1, -1, 15, 24, 15, -1, 8, -1, 13, -1, 13, -1, 11, -1, -1, -1, -1, 0, 0],
                        [2, 2, 19, 14, 24, 1, 15, 19, -1, 21, -1, 2, -1, 24, -1, 3, -1, 2, 1, -1, -1, -1, -1, 0]
                    ])
                elif rate == '5/6':
                    proto = np.array([
                        [17, 13, 8, 21, 9, 3, 18, 12, 10, 0, 4, 15, 19, 2, 5, 10, 26, 19, 13, 13, 1, 0, -1, -1],
                        [3, 12, 11, 14, 11, 25, 5, 18, 0, 9, 2, 26, 26, 10, 24, 7, 14, 20, 4, 2, -1, 0, 0, -1],
                        [22, 16, 4, 3, 10, 21, 12, 5, 21, 14, 19, 5, -1, 8, 5, 18, 11, 5, 5, 15, 0, -1, 0, 0],
                        [7, 7, 14, 14, 4, 16, 16, 24, 24, 10, 1, 7, 15, 6, 10, 26, 8, 18, 21, 14, 1, -1, -1, 0]
                    ])
                else:
                    raise NameError('802.11n invalid rate')
            elif z == 54:
                # N = 1296
                if rate == '1/2':
                    proto = np.array([
                        [40, -1, -1, -1, 22, -1, 49, 23, 43, -1, -1, -1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [50, 1, -1, -1, 48, 35, -1, -1, 13, -1, 30, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [39, 50, -1, -1, 4, -1, 2, -1, -1, -1, -1, 49, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
                        [33, -1, -1, 38, 37, -1, -1, 4, 1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
                        [45, -1, -1, -1, 0, 22, -1, -1, 20, 42, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1],
                        [51, -1, -1, 48, 35, -1, -1, -1, 44, -1, 18, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
                        [47, 11, -1, -1, -1, 17, -1, -1, 51, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
                        [5, -1, 25, -1, 6, -1, 45, -1, 13, 40, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
                        [33, -1, -1, 34, 24, -1, -1, -1, 23, -1, -1, 46, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1],
                        [1, -1, 27, -1, 1, -1, -1, -1, 38, -1, 44, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
                        [-1, 18, -1, -1, 23, -1, -1, 8, 0, 35, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
                        [49, -1, 17, -1, 30, -1, -1, -1, 34, -1, -1, 19, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]
                    ])
                elif rate == '2/3':
                    proto = np.array([
                        [39, 31, 22, 43, -1, 40, 4, -1, 11, -1, -1, 50, -1, -1, -1, 6, 1, 0, -1, -1, -1, -1, -1, -1],
                        [25, 52, 41, 2, 6, -1, 14, -1, 34, -1, -1, -1, 24, -1, 37, -1, -1, 0, 0, -1, -1, -1, -1, -1],
                        [43, 31, 29, 0, 21, -1, 28, -1, -1, 2, -1, -1, 7, -1, 17, -1, -1, -1, 0, 0, -1, -1, -1, -1],
                        [20, 33, 48, -1, 4, 13, -1, 26, -1, -1, 22, -1, -1, 46, 42, -1, -1, -1, -1, 0, 0, -1, -1, -1],
                        [45, 7, 18, 51, 12, 25, -1, -1, -1, 50, -1, -1, 5, -1, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1],
                        [35, 40, 32, 16, 5, -1, -1, 18, -1, -1, 43, 51, -1, 32, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
                        [9, 24, 13, 22, 28, -1, -1, 37, -1, -1, 25, -1, -1, 52, -1, 13, -1, -1, -1, -1, -1, -1, 0, 0],
                        [32, 22, 4, 21, 16, -1, -1, -1, 27, 28, -1, 38, -1, -1, -1, 8, 1, -1, -1, -1, -1, -1, -1, 0]
                    ])
                elif rate == '3/4':
                    proto = np.array([
                        [39, 40, 51, 41, 3, 29, 8, 36, -1, 14, -1, 6, -1, 33, -1, 11, -1, 4, 1, 0, -1, -1, -1, -1],
                        [48, 21, 47, 9, 48, 35, 51, -1, 38, -1, 28, -1, 34, -1, 50, -1, 50, -1, -1, 0, 0, -1, -1, -1],
                        [30, 39, 28, 42, 50, 39, 5, 17, -1, 6, -1, 18, -1, 20, -1, 15, -1, 40, -1, -1, 0, 0, -1, -1],
                        [29, 0, 1, 43, 36, 30, 47, -1, 49, -1, 47, -1, 3, -1, 35, -1, 34, -1, 0, -1, -1, 0, 0, -1],
                        [1, 32, 11, 23, 10, 44, 12, 7, -1, 48, -1, 4, -1, 9, -1, 17, -1, 16, -1, -1, -1, -1, 0, 0],
                        [13, 7, 15, 47, 23, 16, 47, -1, 43, -1, 29, -1, 52, -1, 2, -1, 53, -1, 1, -1, -1, -1, -1, 0]
                    ])
                elif rate == '5/6':
                    proto = np.array([
                        [48, 29, 37, 52, 2, 16, 6, 14, 53, 31, 34, 5, 18, 42, 53, 31, 45, -1, 46, 52, 1, 0, -1, -1],
                        [17, 4, 30, 7, 43, 11, 24, 6, 14, 21, 6, 39, 17, 40, 47, 7, 15, 41, 19, -1, -1, 0, 0, -1],
                        [7, 2, 51, 31, 46, 23, 16, 11, 53, 40, 10, 7, 46, 53, 33, 35, -1, 25, 35, 38, 0, -1, 0, 0],
                        [19, 48, 41, 1, 10, 7, 36, 47, 5, 29, 52, 52, 31, 10, 26, 6, 3, 2, -1, 51, 1, -1, -1, 0]
                    ])
                else:
                    raise NameError('802.11n invalid rate')
            elif z == 81:
                # N = 1944
                if rate == '1/2':
                    proto = np.array([
                        [57, -1, -1, -1, 50, -1, 11, -1, 50, -1, 79, -1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [3, -1, 28, -1, 0, -1, -1, -1, 55, 7, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [30, -1, -1, -1, 24, 37, -1, -1, 56, 14, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
                        [62, 53, -1, -1, 53, -1, -1, 3, 35, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
                        [40, -1, -1, 20, 66, -1, -1, 22, 28, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1],
                        [0, -1, -1, -1, 8, -1, 42, -1, 50, -1, -1, 8, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
                        [69, 79, 79, -1, -1, -1, 56, -1, 52, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
                        [65, -1, -1, -1, 38, 57, -1, -1, 72, -1, 27, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
                        [64, -1, -1, -1, 14, 52, -1, -1, 30, -1, -1, 32, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1],
                        [-1, 45, -1, 70, 0, -1, -1, -1, 77, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
                        [2, 56, -1, 57, 35, -1, -1, -1, -1, -1, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
                        [24, -1, 61, -1, 60, -1, -1, 27, 51, -1, -1, 16, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]
                    ])
                elif rate == '2/3':
                    proto = np.array([
                        [61, 75, 4, 63, 56, -1, -1, -1, -1, -1, -1, 8, -1, 2, 17, 25, 1, 0, -1, -1, -1, -1, -1, -1],
                        [56, 74, 77, 20, -1, -1, -1, 64, 24, 4, 67, -1, 7, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
                        [28, 21, 68, 10, 7, 14, 65, -1, -1, -1, 23, -1, -1, -1, 75, -1, -1, -1, 0, 0, -1, -1, -1, -1],
                        [48, 38, 43, 78, 76, -1, -1, -1, -1, 5, 36, -1, 15, 72, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
                        [40, 2, 53, 25, -1, 52, 62, -1, 20, -1, -1, 44, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1],
                        [69, 23, 64, 10, 22, -1, 21, -1, -1, -1, -1, -1, 68, 23, 29, -1, -1, -1, -1, -1, -1, 0, 0, -1],
                        [12, 0, 68, 20, 55, 61, -1, 40, -1, -1, -1, 52, -1, -1, -1, 44, -1, -1, -1, -1, -1, -1, 0, 0],
                        [58, 8, 34, 64, 78, -1, -1, 11, 78, 24, -1, -1, -1, -1, -1, 58, 1, -1, -1, -1, -1, -1, -1, 0]
                    ])
                elif rate == '3/4':
                    proto = np.array([
                        [48, 29, 28, 39, 9, 61, -1, -1, -1, 63, 45, 80, -1, -1, -1, 37, 32, 22, 1, 0, -1, -1, -1, -1],
                        [4, 49, 42, 48, 11, 30, -1, -1, -1, 49, 17, 41, 37, 15, -1, 54, -1, -1, -1, 0, 0, -1, -1, -1],
                        [35, 76, 78, 51, 37, 35, 21, -1, 17, 64, -1, -1, -1, 59, 7, -1, -1, 32, -1, -1, 0, 0, -1, -1],
                        [9, 65, 44, 9, 54, 56, 73, 34, 42, -1, -1, -1, 35, -1, -1, -1, 46, 39, 0, -1, -1, 0, 0, -1],
                        [3, 62, 7, 80, 68, 26, -1, 80, 55, -1, 36, -1, 26, -1, 9, -1, 72, -1, -1, -1, -1, -1, 0, 0],
                        [26, 75, 33, 21, 69, 59, 3, 38, -1, -1, -1, 35, -1, 62, 36, 26, -1, -1, 1, -1, -1, -1, -1, 0]
                    ])
                elif rate == '5/6':
                    proto = np.array([
                        [13, 48, 80, 66, 4, 74, 7, 30, 76, 52, 37, 60, -1, 49, 73, 31, 74, 73, 23, -1, 1, 0, -1, -1],
                        [69, 63, 74, 56, 64, 77, 57, 65, 6, 16, 51, -1, 64, -1, 68, 9, 48, 62, 54, 27, -1, 0, 0, -1],
                        [51, 15, 0, 80, 24, 25, 42, 54, 44, 71, 71, 9, 67, 35, -1, 58, -1, 29, -1, 53, 0, -1, 0, 0],
                        [16, 29, 36, 41, 44, 56, 59, 37, 50, 24, -1, 65, 4, 65, 52, -1, 4, -1, 73, 52, 1, -1, -1, 0]
                    ])
                else:
                    raise NameError('802.11n invalid rate')
            else:
                raise NameError('802.11n invalid z (must be 27,54 or 81)')
        else:
            raise NameError('IEEE standard unknown')
        return proto

    def pcmat(self):
        """ Converts from a protograph to an LDPC parity-check matrix.
        This function is not used in the live system but is made available
        e.g. if one wants to visualise the actual parity-check matrix.

        Returns
        -------
        np.ndarray
        Parity-check matrix for the LDPC code
        """

        # traverses protograph row/column-wise and assigns all-zero submatrices
        # where the protograph entry is -1, or suitably cyclic-shifted zxz identity
        # matrices where the entry is not -1. Note that use of "np.roll" which
        # operates a cyclic shift of the columns by proto[row,col]%z, and note
        # that the mod z at the end is merely cosmetic since np.roll will
        # natively cyclic shift modulo z if asked to roll a matrix by a shift
        # that exceeds the matrix dimensions.
        proto = self.proto
        z = self.z
        pcmat = np.zeros((z * len(proto), z * len(proto[0])), dtype=int)
        (row, col) = np.nonzero(proto != -1)
        for j in range(len(row)):
            pcmat[row[j] * z:row[j] * z + z, col[j] * z:col[j] * z + z] = np.roll(np.eye(z), proto[row[j], col[j]] % z,
                                                                                  1)

        return pcmat

    def prepare_decoder(self):
        """ Generates the elements required for the LDPC decoder from the
        protograph.

        Parameters
        ----------
        proto: array
        Specifies the protograph for the code.

        z: int
        Specifies the expansion factor for the protograph

        Returns
        -------
        np.array
        vdeg vector of variable node degrees

        np.array
        cdeg vector of constraint node degrees

        np.array
        intrlv vector specifies the interleaver between variable node messages
        and constraint node messages (from a variable node perspective)
        The messages are assumed ordered as required for constraint node processing
        (which is the "harder" processing) and must be addressed through this
        interleaver when processing variable nodes (which is the "easier" processing)
        """

        proto = self.proto
        z = self.z

        # This method operates by assigning interleaver entries and "flagging" them
        # as it traverses the parity-check matrix, so that later visits to the same
        # variable (or constraint) node know to move on to the next available message
        # connection ("port") in the node.

        # Variable node degrees and constraint node degrees are expanded from the
        # "degrees" in the protograph by a factor of z. Note that each column in
        # the protograph results in z columns of the same degree in the actual code,
        # and the same for rows.
        cdeg = np.repeat(np.sum(proto != -1, 1), z)
        vdeg = np.repeat(np.sum(proto != -1, 0), z)
        # Cumulative degrees with a 0 inserted at the start because we need the
        # cumulation "up to and NOT including" the degree of the current node,
        # whereas numpy's cumsum gives us the degree "up to and including"
        # Note that cumvdeg and cumcdeg will be one element too long than
        # we need (we will never use the last entry)
        cumcdeg = np.insert(np.cumsum(cdeg), 0, 0)
        cumvdeg = np.insert(np.cumsum(vdeg), 0, 0)
        # Initialise the interleaver and a vector of flags telling us which "ports"
        # have been used for the constraint nodes, i.e., which messages in each
        # constraint nodes have already been assigned. We also need such flags for
        # the variable node side, but the interleaver doubles up as a flag since
        # we initialised it as -1s, we know that any message that still has a -1
        # is an unuseed port for a variable node.
        intrlv = -np.ones(np.sum(cdeg), dtype=int)
        vflag = np.zeros(np.sum(cdeg), dtype=bool)
        # We will traverse the protograph stopping at each sub-graph that doesn't have
        # a -1 in the protograph (the -1s in the protograph correspond to an all-zero
        # submatrix in the parity-check matrix.
        (xp, yp) = np.nonzero(proto != -1)
        for j in range(xp.size):
            # offset specifies the exponent of the permutation matrix that is inserted
            # at this position in the protograph. An offset of 0 means that the matrix
            # is an identity matrix, whereas an offset of +1 means that the matrix is a
            # "shift one to the right" permutation matrix, etc. Offsets larger than z
            # result in shifts modulo z.
            offset = proto[xp[j], yp[j]]
            for k in range(z):
                # Determine the variable node and constraint node index from the index
                # of the protograph position and the index k of the row/column within
                # the zxz submatrix at this position in the protograph
                cind = xp[j] * z + k
                vind = yp[j] * z + (k + offset) % z
                # Find an unused "port" for the message in the constraint node
                for xi in range(cumcdeg[cind], cumcdeg[cind + 1]):
                    if intrlv[xi] == -1:
                        break
                # Error handling if no unused port found, should never occur
                if intrlv[xi] != -1:
                    raise NameError('No unused port found in constraint node')
                # Find an unused "port" for the message in the variable node
                for yi in range(cumvdeg[vind], cumvdeg[vind + 1]):
                    if vflag[yi] == 0:
                        break
                # Error handling if no unused port found, should never occur
                if vflag[yi] != 0:
                    raise NameError('No unused port found in variable node')
                # now assign the interleaver entry and flag the constraint node "port"
                vflag[yi] = 1
                intrlv[xi] = yi
        intrlv = np.argsort(intrlv)
        return vdeg, cdeg, intrlv

    def encode(self, info):

        z = self.z
        proto = self.proto
        # check dimensions before starting
        Np = len(proto[0])
        N = Np * z  # length of codeword
        Mp = len(proto)
        Kp = Np - Mp
        K = Kp * z  # length of information
        if len(info) != K:
            raise NameError('information word length not compatible with proto and z')

        # x is the codeword, composed of K bits information and N-K bits parity
        x = np.zeros(N, dtype=int)
        x[0:K] = info  # pre-fill the first K bits with the information

        # for the encoding, we will address x z bits at a time, so we reshape it to
        # be Np x z and the rows of x are our new "super-symbols"
        x = np.reshape(x, (Np, z))
        # the following p will contain sum_k x_k H_jk for each row of the prototype parity
        # check matrix, where the sum is only over the systematic (information) part
        p = np.zeros((Mp, z), dtype=int)
        for j in range(Mp):
            ind = np.nonzero(proto[j, 0:Kp] != -1)[0]
            for k in ind.tolist():
                p[j] = np.add(p[j], np.roll(x[k], -proto[j, k]))
        p = np.mod(p, 2)
        tp = np.mod(np.sum(p, 0), 2)  # tp is the sum of the p's
        # The sum of all the super parity-check (vector) equations gives an equation that
        # has only information symbols and the first parity symbol. Most protographs were
        # designed so that the coefficient of the parity symbol in that equation is the
        # identity matrix, but there are a few exceptions where the coefficient is not
        # an identity. The following few lines compute that coefficient and compute its
        # inverse.
        toff = np.zeros(z, dtype=int)
        ind = np.nonzero(proto[:, Kp] != -1)[0]
        for j in ind.tolist():
            toff[proto[j, Kp] % z] += 1
        toff = np.mod(toff, 2)
        tnz = np.nonzero(toff)[0]
        # the coefficients in proto in column Kp come in pairs except one coefficient,
        # resulting in a single coefficient for the first parity symbol. If this is
        # not the case, call an error.
        if len(tnz) != 1:
            raise NameError('The offsets in colum Kp+1 of proto do not add to a single offset')
        toff = tnz[0]
        # now compute the first parity symbol as tp times the inverse coefficient
        # (which will be an offset by 0 in most cases, when the resulting coefficient is
        # an identity matrix)
        x[Kp] = np.roll(tp, toff)
        # the remaining parity symbols are computed using one parity equation at a time
        for j in range(Mp - 1):
            myk = Kp + j + 1  # parity symbol to be computed
            x[myk] = p[j]  # initialise with value of acumulated systematic part
            ind = np.nonzero(proto[j, Kp:myk] != -1)[0]  # search for remaining coefficients
            for k in ind.tolist():
                x[myk] = np.add(x[myk], np.roll(x[Kp + k], -proto[j, Kp + k]))
        x = np.mod(x, 2)

        return (np.reshape(x, -1))

    # ---------------------------
    # ----  C library loading & wrappers ----
    # ---------------------------
    def _load_clib(self):
        """
        Load c_ldpc dynamic library from bin/, set argtypes/restype for functions.
        Supports c_ldpc.pyd (Python extension), c_ldpc.dll (Windows), c_ldpc.so (Linux).
        """
        # determine project root (assume this file is in py/ subdir)
        this_file = os.path.abspath(__file__)
        root_dir = os.path.dirname(os.path.dirname(this_file))
        bin_dir = os.path.join(root_dir, 'bin')

        # candidate names in preferred order
        candidates = ['c_ldpc.pyd', 'c_ldpc.dll', 'c_ldpc.so']
        found = None
        for name in candidates:
            p = os.path.join(bin_dir, name)
            if os.path.exists(p):
                found = p
                break
        if found is None:
            # final attempt: look for any c_ldpc.* in bin
            try:
                for f in os.listdir(bin_dir):
                    if f.startswith('c_ldpc.'):
                        found = os.path.join(bin_dir, f)
                        break
            except Exception:
                pass

        if found is None:
            raise FileNotFoundError("Cannot find c_ldpc library in bin/. Expected names: " + ", ".join(candidates))

        # Load library via ctypes
        # On Windows, CDLL will load .dll/.pyd; on Linux .so
        try:
            self._clib = ct.CDLL(found)
        except Exception as e:
            raise OSError(f"Failed to load LDPC C library '{found}': {e}")

        # Now set function signatures based on src/c_ldpc.c
        # int sumprod(double *ch, long *vdeg, long *cdeg, long *intrlv,
        #            int Nv, int Nc, int Nmsg, double *app)
        self._clib.sumprod.argtypes = [
            ct.POINTER(ct.c_double),  # ch
            ct.POINTER(ct.c_long),  # vdeg
            ct.POINTER(ct.c_long),  # cdeg
            ct.POINTER(ct.c_long),  # intrlv
            ct.c_int, ct.c_int, ct.c_int,
            ct.POINTER(ct.c_double)  # app
        ]
        self._clib.sumprod.restype = ct.c_int

        # sumprod2 has the same signature
        self._clib.sumprod2.argtypes = self._clib.sumprod.argtypes
        self._clib.sumprod2.restype = ct.c_int

        # minsum has an extra double correction_factor at the end
        # int minsum(..., double correction_factor)
        self._clib.minsum.argtypes = [
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_long),
            ct.POINTER(ct.c_long),
            ct.POINTER(ct.c_long),
            ct.c_int, ct.c_int, ct.c_int,
            ct.POINTER(ct.c_double),
            ct.c_double
        ]
        self._clib.minsum.restype = ct.c_int

        # Lxor and Lxfb
        # double Lxor(double L1, double L2, int correction_flag);
        self._clib.Lxor.argtypes = [ct.c_double, ct.c_double, ct.c_int]
        self._clib.Lxor.restype = ct.c_double

        # double Lxfb(double *L, long dc, int correction_flag);
        self._clib.Lxfb.argtypes = [ct.POINTER(ct.c_double), ct.c_long, ct.c_int]
        self._clib.Lxfb.restype = ct.c_double

        return

    def _make_clong_array(self, arr):
        """
        Helper: convert a 1D iterable of integers to a ctypes array of c_long.
        Returns the ctypes array (which must remain alive while C is using it).
        """
        L = len(arr)
        Cls = ct.c_long * L
        # ensure python ints
        return Cls(*[int(x) for x in arr])

    # ======== DGL/CUDA-accelerated Sum-Product (flooding) ========
    def _ensure_dgl_graph(self):
        """首次调用时基于 self.pcmat() 构建 DGL Tanner 图和稀疏 H，并缓存。"""
        if getattr(self, '_dgl_built', False):
            return
        try:
            import torch, dgl
            from scipy.sparse import coo_matrix, csr_matrix
        except Exception as e:
            raise ImportError("使用 'sumprod2_dgl' 需要安装 torch、dgl(含CUDA)、scipy: %s" % e)

        # 默认参数（可外部覆盖）
        self.dgl_device = getattr(self, 'dgl_device', 'cuda')
        self.dgl_llr_clip = float(getattr(self, 'dgl_llr_clip', 20.0))
        self.dgl_max_iter = int(getattr(self, 'dgl_max_iter', 50))
        self.dgl_verbose = bool(getattr(self, 'dgl_verbose', False))
        self.dgl_log_every = int(getattr(self, 'dgl_log_every', 1))
        self.dgl_check_every = int(getattr(self, 'dgl_check_every', 1))
        self.dgl_dtype = getattr(self, 'dgl_dtype', 'float32')
        self.dgl_microbatch = getattr(self, 'dgl_microbatch', None)
        self.dgl_stop_all = getattr(self, 'dgl_stop_all', True)

        self.dgl_dtype = getattr(self, 'dgl_dtype', 'float32')
        self.dgl_microbatch = getattr(self, 'dgl_microbatch', None)
        self.dgl_stop_all = getattr(self, 'dgl_stop_all', True)

        if str(self.dgl_device).startswith('cuda'):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA 不可用，但请求了 GPU 解码（device='cuda'）。")

        # 构建 H（M x N）
        from scipy.sparse import coo_matrix
        H = coo_matrix(self.pcmat()).tocoo()
        M, N = H.shape

        import torch, dgl
        dev = self.dgl_device
        # heterograph，保持 COO 顺序用于边配对
        ci = torch.tensor(H.row, dtype=torch.int64, device=dev)  # check indices
        vi = torch.tensor(H.col, dtype=torch.int64, device=dev)  # var indices
        g = dgl.heterograph({
            ('var', 'v2c', 'check'): (vi, ci),
            ('check', 'c2v', 'var'): (ci, vi),
        }, num_nodes_dict={'var': N, 'check': M}).to(dev)

        # 预分配边特征
        E = g.num_edges(etype='v2c')
        g.edges['v2c'].data['m'] = torch.empty((E, 0), device=dev)
        g.edges['v2c'].data['peer'] = torch.empty((E, 0), device=dev)
        g.edges['c2v'].data['m'] = torch.empty((E, 0), device=dev)
        g.edges['c2v'].data['peer'] = torch.empty((E, 0), device=dev)

        # 稀疏 H（CSR）用于 syndrome
        from scipy.sparse import csr_matrix
        Hcsr = H.tocsr()
        crow = torch.tensor(Hcsr.indptr, dtype=torch.int64, device=dev)
        ccol = torch.tensor(Hcsr.indices, dtype=torch.int64, device=dev)
        data = torch.ones_like(ccol, dtype=torch.float32, device=dev)
        H_csr = torch.sparse_csr_tensor(crow, ccol, data, size=(M, N), device=dev)

        self._dgl_built = True
        self._dgl = {'g': g, 'H_csr': H_csr, 'M': M, 'N': N, 'device': dev}

    def _decode_sumprod2_dgl(self, ch):
        """
        DGL/CUDA 加速的 flooding-schedule sum-product 解码。
        支持 ch 形状：
          - (N,)  单码字
          - (B,N) 批量码字（建议）
        返回：
          - app: (N,) 或 (B,N) 的后验 LLR
          - iters: 实际迭代轮数（批量时为整型，表示统一迭代到全部通过或到达上限）
        """
        import numpy as np
        import torch, dgl

        self._ensure_dgl_graph()
        g = self._dgl['g']
        H_csr = self._dgl['H_csr']
        dev = self._dgl['device']
        LCLIP = self.dgl_llr_clip
        MAXIT = self.dgl_max_iter
        LOGEV = self.dgl_log_every
        CHKEV = max(1, self.dgl_check_every)
        EPS = 1e-12
        _dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}[str(self.dgl_dtype)]

        ch = np.asarray(ch, dtype=np.double)
        was_1d = (ch.ndim == 1)
        if was_1d:
            ch = ch.reshape(1, -1)
        if ch.ndim != 2 or ch.shape[1] != self.Nv:
            raise NameError('Channel input must be shape (N,) or (B,N) 与码长不匹配')

        B = ch.shape[0]
        micro = int(self.dgl_microbatch) if (self.dgl_microbatch or 0) > 0 else B
        app_out = np.zeros((B, self.Nv), dtype=np.float64)
        syn_out = np.zeros((B,1), dtype=np.int32)
        last_iters = 0

        for s in range(0, B, micro):
            e = min(s + micro, B)
            ch_chunk = ch[s:e]
            b = ch_chunk.shape[0]

            E = g.num_edges(etype='v2c')
            g.edges['v2c'].data['m'] = torch.zeros((E, b), dtype=_dtype, device=dev)
            g.edges['v2c'].data['peer'] = torch.zeros((E, b), dtype=_dtype, device=dev)
            g.edges['c2v'].data['m'] = torch.zeros((E, b), dtype=_dtype, device=dev)
            g.edges['c2v'].data['peer'] = torch.zeros((E, b), dtype=_dtype, device=dev)

            L_ch = torch.as_tensor(ch_chunk, dtype=_dtype, device=dev).t().clamp(-LCLIP, LCLIP)
            g.nodes['var'].data['L_ch'] = L_ch

            def init_v2c(edges):
                return {'m': torch.clamp(edges.src['L_ch'], min=-LCLIP, max=LCLIP)}

            g.apply_edges(init_v2c, etype='v2c')

            def cn_reduce(nodes):
                m = nodes.mailbox['m']  # [nodes, deg, b]
                t = torch.tanh(0.5 * torch.abs(m).clamp(max=LCLIP)).clamp(min=EPS)
                prod_t = torch.prod(t, dim=1)  # [nodes,b]
                sgn = torch.sign(torch.where(m == 0, torch.ones_like(m), m))
                prod_s = torch.prod(sgn, dim=1)  # [nodes,b]
                return {'prod_t': prod_t, 'prod_s': prod_s}

            def c2v_apply(edges):
                peer = edges.data['peer']  # [E,b]
                t_peer = torch.tanh(0.5 * torch.abs(peer).clamp(max=LCLIP)).clamp(min=EPS)
                prod_ex = torch.clamp(edges.src['prod_t'] / t_peer, min=EPS, max=1.0 - 1e-7)
                s_ex = edges.src['prod_s'] * torch.sign(torch.where(peer == 0, torch.ones_like(peer), peer))
                mag = 2.0 * torch.atanh(prod_ex.to(torch.float32))
                out = torch.clamp((s_ex.to(torch.float32) * mag), min=-LCLIP, max=LCLIP).to(_dtype)
                return {'m': out}

            def v2c_apply(edges):
                out = edges.src['L_ch'] + edges.src['sum_c2v'] - edges.data['peer']
                out = torch.clamp(out, min=-LCLIP, max=LCLIP)
                return {'m': out}

            iters = 0
            for it in range(1, MAXIT + 1):
                iters = it
                g['v2c'].update_all(dgl.function.copy_e('m', 'm'), cn_reduce)
                g.edges['c2v'].data['peer'] = g.edges['v2c'].data['m']
                g.apply_edges(c2v_apply, etype='c2v')
                g['c2v'].update_all(dgl.function.copy_e('m', 'm'), dgl.function.sum('m', 'sum_c2v'))
                Lpost = (g.nodes['var'].data['L_ch'].to(torch.float32) + g.nodes['var'].data['sum_c2v'].to(
                    torch.float32))
                g.nodes['var'].data['Lpost'] = Lpost.to(_dtype)
                g.edges['v2c'].data['peer'] = g.edges['c2v'].data['m']
                g.apply_edges(v2c_apply, etype='v2c')

                if (it % CHKEV == 0) or (it == 1) or (it == MAXIT):
                    xhat = (Lpost < 0.0).to(torch.float32)  # [N,b]
                    syn = torch.remainder(torch.sparse.mm(H_csr, xhat), 2.0)  # [M,b]
                    syn_sum = torch.sum(syn, dim=0)  # [b]
                    if self.dgl_verbose and ((it % LOGEV) == 0 or (syn_sum == 0).all() or it in (1, MAXIT)):
                        passed = int((syn_sum == 0).sum().item())
                        print(f"[DGL] iter={it} passed={passed}/{b}")
                    if getattr(self, 'dgl_stop_all', True) and (syn_sum == 0).all():
                        break

            last_iters = iters
            app_np = g.nodes['var'].data['Lpost'].detach().to(torch.float32).t().cpu().numpy().astype(np.float64)  # (b,N)
            app_out[s:e] = app_np
            syn_out[s:e] = syn_sum.cpu().numpy().reshape(-1,1).astype(np.int32)

        if was_1d:
            return app_out[0], int(last_iters), syn_out[0]
        else:
            return app_out, int(last_iters), syn_out

    def decode(self, ch, dectype='sumprod2', corr_factor=0.7):
        """
        ch: numpy array-like of length Nv, dtype float (LLR inputs)
        dectype: 'sumprod', 'sumprod2', 'minsum', 'sumprod2_dgl'
        returns (app (numpy float64 array), iterations (int))
        """
        # basic checks and conversions
        ch = np.asarray(ch, dtype=np.double)

        # --- DGL GPU branch (early exit) ---
        if dectype == 'sumprod2_dgl':
            app, iters, syn = self._decode_sumprod2_dgl(ch)
            if getattr(self, 'print_iter', False):
                print(f"[LDPC][{dectype}] iters={iters}")
            return app, iters, syn

        # ===== CPU/C backend path =====
        if ch.ndim != 1:
            raise NameError('CPU/C 解码仅支持 1D 输入（单码字）。GPU 路径可用批量。')
        if len(ch) != self.Nv:
            raise NameError('Channel inputs not consistent with variable degrees')
        Nv = self.Nv
        Nc = self.Nc
        Nmsg = self.Nmsg

        # prepare app array (numpy, contiguous)
        app = np.zeros(Nv, dtype=np.double)

        # prepare ctypes arrays for integer vectors (c_long)
        vdeg_c = self._make_clong_array(self.vdeg)
        cdeg_c = self._make_clong_array(self.cdeg)
        intrlv_c = self._make_clong_array(self.intrlv)

        # pointers
        ch_p = ch.ctypes.data_as(ct.POINTER(ct.c_double))
        app_p = app.ctypes.data_as(ct.POINTER(ct.c_double))
        vdeg_p = ct.cast(vdeg_c, ct.POINTER(ct.c_long))
        cdeg_p = ct.cast(cdeg_c, ct.POINTER(ct.c_long))
        intrlv_p = ct.cast(intrlv_c, ct.POINTER(ct.c_long))

        # call the requested decoder
        if dectype == 'sumprod':
            it = self._clib.sumprod(ch_p, vdeg_p, cdeg_p, intrlv_p, Nv, Nc, Nmsg, app_p)
        elif dectype == 'sumprod2':
            it = self._clib.sumprod2(ch_p, vdeg_p, cdeg_p, intrlv_p, Nv, Nc, Nmsg, app_p)
        elif dectype == 'minsum':
            it = self._clib.minsum(ch_p, vdeg_p, cdeg_p, intrlv_p, Nv, Nc, Nmsg, app_p, ct.c_double(corr_factor))
        else:
            raise NameError('Decoder type unknown')

        return app, int(it)

    # convenience wrappers for Lxor and Lxfb

    def Lxor(self, L1, L2, corrflag=1):
        return float(self._clib.Lxor(ct.c_double(L1), ct.c_double(L2), int(corrflag)))

    def Lxfb(self, L, corrflag=1):
        L = np.asarray(L, dtype=np.double)
        L_p = L.ctypes.data_as(ct.POINTER(ct.c_double))
        dc = ct.c_long(len(L))
        res = float(self._clib.Lxfb(L_p, dc, int(corrflag)))
        return res, L

# End of file