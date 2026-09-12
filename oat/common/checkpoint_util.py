from typing import Optional, Dict
import json
import math
import os

class TopKCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()
    
    def restore_from_logs(self, log_path, next_epoch):
        """Recover rankings for retained, completed checkpoints without deleting files.

        The log supplies full-precision scores; rounded filenames are used only
        to locate checkpoints. Ignore incomplete writes and epochs beyond the
        continuation point, which may belong to an interrupted later attempt.
        """
        self.path_value_map = {}
        if self.k == 0 or not os.path.isfile(log_path):
            return 0
        candidates = {}
        with open(log_path) as stream:
            for line in stream:
                if not line.endswith('\n'):
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                epoch = data.get('epoch')
                value = data.get(self.monitor_key)
                if (type(epoch) is not int or not 0 <= epoch < next_epoch
                        or type(value) not in (int, float) or not math.isfinite(value)):
                    continue
                try:
                    path = os.path.join(self.save_dir, self.format_str.format(**data))
                except (KeyError, ValueError, TypeError):
                    continue
                if os.path.isfile(path):
                    candidates[path] = (value, epoch)
        ranked = sorted(candidates.items(), key=lambda item: (
            item[1][0] if self.mode == 'min' else -item[1][0],
            -item[1][1], item[0]))
        self.path_value_map = {path: value for path, (value, _) in ranked[:self.k]}
        return len(self.path_value_map)

    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None
        if self.monitor_key not in data:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))
        
        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path
        
        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path
