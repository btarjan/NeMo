# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections import defaultdict

from nemo_text_processing.text_normalization.data_loader_utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.graph_utils import NEMO_SIGMA, GraphFst
from nemo_text_processing.text_normalization.ru.taggers.number_names import NumberNamesFst
from nemo_text_processing.text_normalization.ru.taggers.numbers_alternatives import AlternativeFormatsFst

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g. 
        -23 -> cardinal { negative: "true"  integer: "twenty three" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        print('Ru TN only support non-deterministic cases and produces multiple normalization options.')
        n = NumberNamesFst()
        cardinal = n.cardinal_number_names

        alternative_formats = AlternativeFormatsFst()
        one_thousand_alternative = alternative_formats.one_thousand_alternative
        separators = alternative_formats.separators

        cardinal |= cardinal @ one_thousand_alternative
        cardinal_numbers = separators @ cardinal
        self.cardinal_numbers = cardinal_numbers
        final_graph = self.add_tokens(cardinal_numbers)
        self.fst = final_graph.optimize()


if __name__ == '__main__':
    fst = CardinalFst()
    from pynini.lib.rewrite import rewrites

    print(rewrites("2300", fst.graph_default))

    import pdb

    pdb.set_trace()
    print(rewrites("2.300", fst.cardinal_numbers))
    print()

    written = "1696"
    spoken = "тысяча шестьсот девяносто шестом"
