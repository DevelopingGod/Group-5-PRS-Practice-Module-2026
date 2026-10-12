# Notebooks

Colab notebooks for training and experiments. Clear outputs before committing (large outputs and embedded images bloat the repo, and images are blocked by the hygiene check).

Start every notebook with:

```python
from google.colab import drive
drive.mount("/content/drive")
!git clone https://github.com/DevelopingGod/Group-5-PRS-Practice-Module-2026.git bin-buddy && cd bin-buddy && pip install -e contracts
import sys; sys.path.append("/content/bin-buddy/ml")
from binbuddy_ml.labels import load_labels
```
