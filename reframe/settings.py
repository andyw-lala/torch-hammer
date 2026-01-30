# ReFrame configuration for Torch Hammer benchmarks
#
# This configuration works out-of-the-box for local testing.
# For SLURM/PBS clusters, see the commented examples below.
#
# Usage:
#   reframe -C reframe/settings.py -c reframe/torch_hammer_checks.py -r
#
# Documentation: https://reframe-hpc.readthedocs.io/

import os

site_configuration = {
    'systems': [
        # ============================================================
        # Generic local system - works anywhere
        # ============================================================
        {
            'name': 'generic',
            'descr': 'Local GPU testing (no scheduler)',
            'hostnames': ['.*'],  # Match any hostname
            'modules_system': 'nomod',  # No module system
            'partitions': [
                {
                    'name': 'gpu',
                    'descr': 'Local GPU',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'access': [],
                    'environs': ['default'],
                    'max_jobs': 1,
                }
            ]
        },
        
        # ============================================================
        # SLURM cluster template (UNCOMMENT AND CUSTOMIZE)
        # ============================================================
        # Every SLURM cluster is different. Customize these settings:
        #   - hostnames: regex pattern matching your login/compute nodes
        #   - access: partition (-p) and account (-A) flags
        #   - modules: your site's module names
        #   - resources: GPU allocation syntax varies by site
        #
        # {
        #     'name': 'my-slurm-cluster',
        #     'descr': 'My SLURM Cluster',
        #     'hostnames': [r'login\d+', r'node\d+'],
        #     'modules_system': 'lmod',
        #     'partitions': [
        #         {
        #             'name': 'gpu',
        #             'descr': 'GPU partition',
        #             'scheduler': 'slurm',
        #             'launcher': 'srun',
        #             'access': ['-p gpu', '-A myaccount'],
        #             'environs': ['gpu-env'],
        #             'max_jobs': 4,
        #             'resources': [
        #                 {
        #                     'name': 'gpu',
        #                     'options': ['--gres=gpu:{num_gpus}']
        #                 }
        #             ]
        #         }
        #     ]
        # },
        
        # ============================================================
        # PBS cluster template (UNCOMMENT AND CUSTOMIZE)
        # ============================================================
        # {
        #     'name': 'my-pbs-cluster',
        #     'descr': 'My PBS Cluster',
        #     'hostnames': [r'pbs.*'],
        #     'modules_system': 'lmod',
        #     'partitions': [
        #         {
        #             'name': 'gpu',
        #             'descr': 'GPU queue',
        #             'scheduler': 'pbs',
        #             'launcher': 'mpiexec',
        #             'access': ['-q gpu'],
        #             'environs': ['default'],
        #             'max_jobs': 4,
        #         }
        #     ]
        # },
    ],
    
    'environments': [
        # Default environment - no modules needed
        {
            'name': 'default',
            'cc': 'gcc',
            'cxx': 'g++',
            'target_systems': ['*']
        },
        
        # Example: CUDA environment (UNCOMMENT AND CUSTOMIZE)
        # {
        #     'name': 'gpu-env',
        #     'modules': ['cuda/12.4', 'python/3.11'],
        #     'cc': 'gcc',
        #     'cxx': 'g++',
        #     'target_systems': ['my-slurm-cluster']
        # },
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
                    'name': 'reframe.log',
                    'level': 'debug',
                    'format': '[%(asctime)s] %(levelname)s: %(message)s',
                    'append': False
                }
            ],
            'handlers_perflog': [
                {
                    'type': 'filelog',
                    'prefix': '%(check_system)s/%(check_partition)s',
                    'level': 'info',
                    'format': (
                        '%(check_job_completion_time)s|'
                        '%(check_name)s|'
                        '%(check_system)s|'
                        '%(check_partition)s|'
                        '%(check_environ)s|'
                        '%(check_perfvalues)s'
                    ),
                    'append': True
                }
            ]
        }
    ],
    
    'general': [
        {
            'check_search_path': ['reframe/'],
            'check_search_recursive': True,
            'purge_environment': True,
            'resolve_module_conflicts': False
        }
    ]
}
