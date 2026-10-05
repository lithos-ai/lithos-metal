# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Full-width rotary tables, with the Llama 3 frequency-dependent scaling rule.

The rule follows transformers/modeling_rope_utils.py::_compute_llama3_parameters
@ v5.17.0 (Apache-2.0); implemented in NumPy without a transformers dependency.
"""
import numpy as np


def frequencies(config):
    # Round the power once to FP32, matching the HF CPU pow result; NumPy
    # float32 vector power can differ by one ULP and drift at long positions.
    exponent = np.arange(0, config.head_dim, 2, dtype=np.float64) / config.head_dim
    inv = np.float32(1.) / np.power(config.rope_theta, exponent).astype(np.float32)
    p = config.rope_scaling
    if p.get('rope_type', 'default') == 'llama3':
        wavelength = np.float32(2 * np.pi) / inv
        low, high = p['low_freq_factor'], p['high_freq_factor']
        original, factor = p['original_max_position_embeddings'], p['factor']
        blend = np.clip((original / wavelength - low) / (high - low), 0., 1.)
        inv = (1. - blend) * (inv / factor) + blend * inv
    return inv.astype(np.float32)


def tables(config, positions):
    phase = np.outer(np.asarray(positions, np.float32), frequencies(config))
    phase = np.concatenate((phase, phase), axis=-1)
    return np.cos(phase).astype(np.float32), np.sin(phase).astype(np.float32)
