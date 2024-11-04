import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supervisely.nn.benchmark.comparison.sem_seg_model_comparison import (
    SemanticSegmentationComparison,
)


ssc = SemanticSegmentationComparison(eval_dirs=["output", "output_2"])
ssc.create_comparison_charts()
