"""Override the inherited inclusive selector only for the HScore family."""
from .selection import SELECTION_TAG, selection_metadata


class StrictSelection:
    def selection_protocol(self):
        return dict(super().selection_protocol(), selection=SELECTION_TAG,
                    checkpoint_selection=selection_metadata())

    @property
    def output_version(self):
        return super().output_version.replace("select_sr092_", "select_srgt092_")

    def normalization_metadata(self):
        return dict(super().normalization_metadata(), checkpoint_selection=selection_metadata())

    def reproducibility_metadata(self):
        return dict(super().reproducibility_metadata(), checkpoint_selection=selection_metadata())

    def evaluation_csv(self):
        path = super().evaluation_csv()
        return path.with_name(path.name.replace("_sr092_", "_srgt092_"))

    def describe(self, algorithm=None):
        return dict(super().describe(algorithm), checkpoint_selection=selection_metadata())
