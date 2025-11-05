import os
import glob
import torch
import pyarrow as pa

import torch
import torch.utils

class DistributedDataset(torch.utils.data.IterableDataset):
    def __init__(self, root_dir, rank, world_size, batch_size, seq_length, bos_token, eos_token, reverse=False):
        super().__init__()
        self.root_dir = root_dir
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.bos_token = bos_token
        self.eos_token = eos_token

        total_num_arrow_files = len(glob.glob(os.path.join(self.root_dir, "rank_*.arrow")))
        print('world size:', self.world_size)
        # assert total_num_arrow_files % self.world_size == 0
        self.num_arrow_files_per_rank = total_num_arrow_files // self.world_size
        self.arrow_readers = [pa.ipc.open_file(pa.memory_map(os.path.join(self.root_dir, f"rank_{self.rank + i * self.world_size}.arrow"))) for i in range(self.num_arrow_files_per_rank)]

        self.reverse = reverse
        self.buffer = []
        self.current_arrow_idx = len(self.arrow_readers) - 1 if self.reverse else 0
        self.current_batch_idx = 0

    def __iter__(self):
        if self.reverse:
            arrow_indices = range(self.current_arrow_idx, -1, -1)
        else:
            arrow_indices = range(self.current_arrow_idx, len(self.arrow_readers))
            
        for arrow_idx in arrow_indices:
            self.current_arrow_idx = arrow_idx
            arrow_reader = self.arrow_readers[arrow_idx]
            
            batch_indices = range(self.current_batch_idx, arrow_reader.num_record_batches)
            # if self.reverse:
            #     batch_indices = reversed(batch_indices)
            # NOTE don't have to reverse batch idx
            
            for batch_idx in batch_indices:
                self.current_batch_idx = batch_idx
                sample = arrow_reader.get_batch(batch_idx)['input_ids'].to_pylist()
                self.buffer += [self.bos_token] + sample + [self.eos_token]
                while len(self.buffer) >= self.batch_size * self.seq_length + 1:
                    yield torch.LongTensor(self.buffer[:self.seq_length * self.batch_size]).reshape(self.batch_size, self.seq_length), \
                          torch.LongTensor(self.buffer[1:self.seq_length * self.batch_size + 1]).reshape(self.batch_size, self.seq_length)
                    self.buffer = self.buffer[self.seq_length * self.batch_size:]

            self.current_batch_idx = 0

    def state_dict(self):
        """Return a dictionary containing the state of the dataset."""
        return {
            'buffer': self.buffer,
            'current_arrow_idx': self.current_arrow_idx,
            'current_batch_idx': self.current_batch_idx,
        }

    def load_state_dict(self, state_dict):
        """Load the state of the dataset."""
        self.buffer = state_dict['buffer']
        self.current_arrow_idx = state_dict['current_arrow_idx']
        self.current_batch_idx = state_dict['current_batch_idx']