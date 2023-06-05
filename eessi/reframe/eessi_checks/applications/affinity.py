"""
This module tests binding affinity for hybrid workloads
"""

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher

from eessi_utils import hooks, utils
from eessi_utils.constants import DEVICES, FEATURES, SCALES, TAGS

@rfm.simple_test
class AFFINITY_EESSI(rfm.RunOnlyRegressionTest):
    """
    This test simulates a hybrid MPI program.
    We don't check the actual binding, since it depends on node topology.
    However, the stdout produced by this test can easily be manually inspected
    to validate the sanity of the binding pattern.
    """

    # This test can run at any scale, so parameterize over all known SCALES
    scale = parameter(SCALES.keys())
    valid_prog_environs = ['default']
    valid_systems = []

    # Parameterize over all modules that start with TensorFlow
    module_name = parameter(utils.find_modules('affinity'))

    # Make CPU and GPU versions of this test
    device_type = parameter([DEVICES['CPU'], DEVICES['GPU']])

    # Test srun and mpirun as launchers
    launcher = parameter(['srun', 'mpirun'])

    executable = 'affinity'

    time_limit = '5m'

    # This test should be run as part of EESSI CI
    tags = {TAGS['CI']}

    @sanity_function
    def assert_no_error(self):
        """Check that the error file is completely empty"""
        return sn.assert_found(r'\A^$\Z', self.stderr)

    @run_after('init')
    def run_after_init(self):
        """hooks to run after the init phase"""
        hooks.set_modules(self)
        hooks.set_tag_scale(self)
        # Can't use hooks.filter_valid_systems_by_device_type, because it will check if the module has CUDA support
        # affinity doesn't, but does not need to. We want to run it on host CPU, even on GPU nodes
        if self.device_type == DEVICES['CPU']:
            self.valid_systems = [f'+{FEATURES["CPU"]}']
        elif self.device_type == DEVICES['GPU']:
            self.valid_systems = [f'+{FEATURES["GPU"]}']
        else:
            raise NotImplementedError(f'Failed to set valid partition for device type {self.device_type}')

    @run_after('init')
    def set_test_descr(self):
        self.descr = f'Affinity benchmark on {self.device_type}'

    @run_after('setup')
    def run_after_setup(self):
        """hooks to run after the setup phase"""
        if self.device_type == DEVICES['CPU']:
            hooks.assign_one_task_per_compute_unit(test=self, compute_unit=DEVICES['CPU_SOCKET'])
        elif self.device_type == DEVICES['GPU']:
            hooks.assign_one_task_per_compute_unit(test=self, compute_unit=DEVICES['GPU'])
        else:
            raise NotImplementedError(f'Failed to set number of tasks and cpus per task for device {self.device_type}')

    @run_after('setup')
    def set_binding_policy(self):
        hooks.set_binding_policy(self)

    @run_after('setup')
    def set_launcher(self):
        self.job.launcher = getlauncher(self.launcher)()

