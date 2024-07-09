class ShiftTimeSeriesSplit:
    def __init__(
        self, 
        length, 
        test_size=1, 
        min_train_size=1, 
        max_train_size=np.inf,
        overlap=False,
    ):
        assert length >= 1, "Sample length must be a positive integer"
        assert test_size >= 1, "Test set size must be a positive integer"
        
        assert min_train_size > 0
        if not max_train_size is None:
            assert max_train_size > 0, "Maximum train set size must be a positive integer"
            assert max_train_size >= min_train_size, "Maximum train set size must be greater or equal to minimum split size"

        assert overlap is True or overlap is False, "Overlap must be a boolean"

        self.length = length

        self.test_size = test_size
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.overlap = overlap
        
    def split(self):
        flag_mss = False
        if not self.max_train_size is None:
            flag_mss = True

        T = self.length
        # Compute number of splits
        if self.overlap:
            n_splits = T - self.min_train_size - self.test_size + 1
        else:
            m = T - self.min_train_size
            n_splits = np.floor(m / self.test_size).astype(int)
            if self.test_size == 1:
                self.n_splits -= 1

        train_idxs = []
        test_idxs  = []
        t_i = self.min_train_size
        for _ in range(n_splits):
            v_i = t_i if self.max_train_size is None else self.max_train_size
            start_idx = max(0, t_i - v_i) if flag_mss else 0

            train_idxs.append(list(range(start_idx, t_i)))
            test_idxs.append(list(range(t_i, t_i + self.test_size)))

            if self.overlap:
                t_i += 1
            else:
                t_i += self.test_size

        return tuple(zip(train_idxs, test_idxs))

    def __iter__(self):
        self.flag_mss = False if not self.max_train_size is None else True

        T = self.length
        # Compute number of splits
        if self.overlap:
            self.n_splits = T - self.min_train_size - self.test_size + 1
        else:
            m = T - self.min_train_size
            self.n_splits = np.floor(m / self.test_size).astype(int)
            if self.test_size == 1:
                self.n_splits -= 1

        # iterator variables
        self.i = 0
        self.minss = self.min_train_size
        self.maxss = self.max_train_size

        return self

    def __next__(self):
        if self.i < self.n_splits:
            v_i = self.minss if self.max_train_size is None else self.max_train_size
            start_idx = max(0, self.minss - v_i) if self.flag_mss else 0

            train_idx = list(range(start_idx, self.minss))
            test_idxs = list(range(self.minss, self.minss + self.test_size))
            
            self.i += 1
            if self.overlap:
                self.minss += 1
            else:
                self.minss += self.test_size

            return (train_idx, test_idxs)
        else:
            raise StopIteration