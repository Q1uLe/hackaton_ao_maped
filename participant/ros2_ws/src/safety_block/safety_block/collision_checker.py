import numpy as np
from csv import DictReader
import mujoco
from math import ceil, degrees


class CollisionChecker:
    def __init__(self, xml_path, csv_path) -> None:
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.pairs = set()
        with open(csv_path) as csv_file:
            for row in DictReader(csv_file):
                if row['diagnostic_policy'] == 'checked':
                    self.pairs.add((
                        int(row['geom1_id']),
                        int(row['geom2_id'])
                    ))
    
    def check_configuration(self, q: np.ndarray) -> bool:
        self.data.qpos[7:] = q
        mujoco.mj_forward(self.model, self.data)
        for contact in self.data.contact:
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            if (geom1_id, geom2_id) in self.pairs or (geom2_id, geom1_id) in self.pairs:
                if contact.dist < -0.001:
                    return False
        return True

    def check_process(self, q_start: np.ndarray, q_end: np.ndarray) -> bool:
        max_q_increment = max(abs(q_end - q_start))
        step_count = ceil(degrees(max_q_increment) / 0.1)
        if step_count == 0:
            return self.check_configuration(q_start)
        step = (q_end - q_start) / step_count
        for i in range(step_count+1):
            pos = q_start + i * step
            if not self.check_configuration(pos):
                return False
        return True

