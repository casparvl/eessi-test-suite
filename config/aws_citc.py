# This is an example configuration file

# Note that CPU autodetect currently does not work with this configuration file on AWS.
# This is because there is no system mpirun, and the CPU autodetection doesn't load any modules
# that would make an mpirun command available (as normal multiprocessing tests would).
# In order to do CPU autodetection, you'll need to change the launcer to srun:
# 'launcher = srun'
# You can run the CPU autodetect by listing all tests (reframe -l ...)
# and then, once all CPUs are autodetected, change the launcher back to mpirun for a 'real' run (reframe -r ...)

# Another known issue is that CPU autodetection fails if run from an actual installation of ReFrame.
# It only works if run from a clone of their Github Repo. See https://github.com/reframe-hpc/reframe/issues/2914

from os import environ, makedirs

from eessi.testsuite.constants import FEATURES

# Get username of current user
homedir = environ.get('HOME')

# This config will write all staging, output and logging to subdirs under this prefix
reframe_prefix = f'{homedir}/reframe_runs'
log_prefix = f'{reframe_prefix}/logs'

# ReFrame complains if the directory for the file logger doesn't exist yet
makedirs(f'{log_prefix}', exist_ok=True)

# AWS CITC site configuration
site_configuration = {
    'systems': [
        {
            'name': 'citc',
            'descr': 'Cluster in the Cloud build and test environment on AWS',
            'modules_system': 'lmod',
    	    'hostnames': ['mgmt', 'login', 'fair-mastodon*'],
            'prefix': reframe_prefix,
            'partitions': [
                {
                    'name': 'x86_64-haswell-8c-15gb',
                    'access': ['--constraint=shape=c4.2xlarge', '--export=NONE'],
                    'descr': 'Haswell, 8 cores, 15 GiB',
                },
                {
                    'name': 'x86_64-haswell-16c-30gb',
                    'access': ['--constraint=shape=c4.4xlarge', '--export=NONE'],
                    'descr': 'Haswell, 16 cores, 30 GiB',
                },
                {
                    'name': 'x86_64-zen2-8c-16gb',
                    'access': ['--constraint=shape=c5a.2xlarge', '--export=NONE'],
                    'descr': 'Zen2, 8 cores, 16 GiB',
                },
                {
                    'name': 'x86_64-zen2-16c-32gb',
                    'access': ['--constraint=shape=c5a.4xlarge', '--export=NONE'],
                    'descr': 'Zen2, 16 cores, 32 GiB',
                },
                {
                    'name': 'x86_64-zen3-8c-16gb',
                    'access': ['--constraint=shape=c6a.2xlarge', '--export=NONE'],
                    'descr': 'Zen3, 8 cores, 16 GiB',
                },
                {
                    'name': 'X86_64-zen3-16c-32gb',
                    'access': ['--constraint=shape=c6a.4xlarge', '--export=NONE'],
                    'descr': 'Zen3, 16 cores, 32 GiB',
                },
                {
                    'name': 'x86_64-skylake-cascadelake-8c-16gb',
                    'access': ['--constraint=shape=c5.2xlarge', '--export=NONE'],
                    'descr': 'Skylake/Cascade lake, 8 cores, 16 GiB',
                },
                {
                    'name': 'x86_64-skylake-cascadelake-16c-32gb',
                    'access': ['--constraint=shape=c5.4xlarge', '--export=NONE'],
                    'descr': 'Skylake/Cascade lake, 16 cores, 32 GiB',
                },
                {
                    'name': 'x86_64-skylake-cascadelake-8c-16gb-nvme',
                    'access': ['--constraint=shape=c5d.2xlarge', '--export=NONE'],
                    'descr': 'Skylake/Cascade lake, 8 cores, 16 GiB, 200GB NVMe',
                },
                {
                    'name': 'x86_64-icelake-8c-16gb',
                    'access': ['--constraint=shape=c6i.2xlarge', '--export=NONE'],
                    'descr': 'Icelake, 8 cores, 16 GiB',
                },
                {
                    'name': 'aarch64-graviton2-8c-16gb',
                    'access': ['--constraint=shape=c6g.2xlarge', '--export=NONE'],
                    'descr': 'Graviton2, 8 cores, 16 GiB',
                },
                {
                    'name': 'aarch64-graviton2-16c-32gb',
                    'access': ['--constraint=shape=c6g.4xlarge', '--export=NONE'],
                    'descr': 'Graviton2, 16 cores, 32 GiB',
                },
                {
                    'name': 'aarch64-graviton2-32c-64gb',
                    'access': ['--constraint=shape=c6g.8xlarge', '--export=NONE'],
                    'descr': 'Graviton2, 32 cores, 64 GiB',
                },
                {
                    'name': 'aarch64-graviton3-8c-16gb',
                    'access': ['--constraint=shape=c7g.2xlarge', '--export=NONE'],
                    'descr': 'Graviton3, 8 cores, 16 GiB',
                },
                {
                    'name': 'aarch64-graviton3-16c-32gb',
                    'access': ['--constraint=shape=c7g.4xlarge', '--export=NONE'],
                    'descr': 'Graviton3, 16 cores, 32 GiB',
                },
             ]
         },
     ],
    'environments': [
        {
            'name': 'default',
            'cc': 'cc',
            'cxx': '',
            'ftn': '',
        },
     ],
     'logging': [
        {
            'level': 'debug',
            'handlers': [
                {
                    'type': 'stream',
                    'name': 'stdout',
                    'level': 'info',
                    'format': '%(message)s'
                },
                {
                    'type': 'file',
                    'prefix': f'{log_prefix}/reframe.log',
                    'name': 'reframe.log',
                    'level': 'debug',
                    'format': '[%(asctime)s] %(levelname)s: %(check_info)s: %(message)s',   # noqa: E501
                    'append': True,
                    'timestamp': "%Y%m%d_%H%M%S",
                },
            ],
            'handlers_perflog': [
                {
                    'type': 'filelog',
                    'prefix': f'{log_prefix}/%(check_system)s/%(check_partition)s',
                    'level': 'info',
                    'format': (
                        '%(check_job_completion_time)s|reframe %(version)s|'
                        '%(check_info)s|jobid=%(check_jobid)s|'
                        '%(check_perf_var)s=%(check_perf_value)s|'
                        'ref=%(check_perf_ref)s '
                        '(l=%(check_perf_lower_thres)s, '
                        'u=%(check_perf_upper_thres)s)|'
                        '%(check_perf_unit)s'
                    ),
                    'append': True
                }
            ]
        }
    ],
    'general': [
        {
            'remote_detect': True,
        }
    ],
}

# Add default things to each partition:
partition_defaults = {
    'scheduler': 'squeue',
    # mpirun causes problems with cpu autodetect, since there is no system mpirun.
    # See https://github.com/EESSI/test-suite/pull/53#issuecomment-1590849226
    # and this feature request https://github.com/reframe-hpc/reframe/issues/2926
    # However, using srun requires either using pmix or proper pmi2 integration in the MPI library
    # See https://github.com/EESSI/test-suite/pull/53#issuecomment-1598753968
    # Thus, we use mpirun for now, and manually swap to srun if we want to autodetect CPUs...
    'launcher': 'srun',
    'environs': ['default'],
    'features': [
        FEATURES['CPU']
    ],
    'prepare_cmds': [
        'source /cvmfs/pilot.eessi-hpc.org/latest/init/bash',
        # Required when using srun as launcher with --export=NONE in partition access, in order to ensure job
        # steps inherit environment. It doesn't hurt to define this even if srun is not used
        'export SLURM_EXPORT_ENV=ALL'
    ],
}
for system in site_configuration['systems']:
    for partition in system['partitions']:
        partition.update(partition_defaults)


