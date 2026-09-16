from pathlib import Path
import tempfile
import unittest
from main_experiment.common import write_json,sha,verify_immutable_checkpoints

class ResumeTests(unittest.TestCase):
    def test_completed_checkpoint_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); p=root/'calibration/g/validation_1_128.json'
            write_json(p,{'volumes':[1,2]})
            write_json(root/'report.json',{'status':'started'})
            write_json(root/'checksums.json',{'calibration/g/validation_1_128.json':sha(p),'report.json':sha(root/'report.json')})
            verify_immutable_checkpoints(root)
            # A mutable status may advance during an interrupted invocation.
            write_json(root/'report.json',{'status':'progress'})
            verify_immutable_checkpoints(root)
            write_json(p,{'volumes':[3,4]})
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                verify_immutable_checkpoints(root)
