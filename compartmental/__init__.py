# Copyright 2023 Unai Lería Fortea

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 
__version__ = "0.1.0"
_CUPY_MODE_: bool

from . import generic_model
from . import util
from . import parameters

def use_cupy():
    import cupy
    CNP = cupy
    _CUPY_MODE_= True
    update_packages(CNP)

def use_numpy():
    import numpy
    CNP = numpy
    _CUPY_MODE_= False
    update_packages(CNP)

def update_packages(CNP):
    generic_model.CNP = CNP
    util.CNP = CNP
    parameters.parameters_manager.CNP = CNP

use_numpy()

from .generic_model import GenericModel