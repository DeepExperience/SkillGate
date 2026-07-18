# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import abc
import os
import random
import socket
from datetime import timedelta

import ray
import torch
import torch.distributed as dist

import relax.utils.training.eval_config
from relax.distributed.ray.ray_actor import RayActor
from relax.utils import device as device_utils
from relax.utils.distributed_utils import init_gloo_group
from relax.utils.misc import get_current_node_ip
from relax.utils.logging_utils import get_logger
from relax.utils.memory_utils import clear_memory, print_memory


logger = get_logger(__name__)


_ROLE_MASTER_PORT_DEFAULTS = {
    # Keep explicit torch TCPStore ports outside Linux's ephemeral range
    # (32768-65000 on the current Ray nodes). NCCL bootstrap sockets also use
    # that kernel range, so putting MASTER_PORT there can race with lazy NCCL
    # communicator initialization during CP=2 training.
    "actor": (20000, 20999),
    "actor_fwd": (21000, 21999),
    "reference": (22000, 22999),
    "critic": (26000, 26999),
    "default": (27000, 29999),
}


def get_local_gpu_id():
    cvd = os.environ.get(device_utils.get_visible_devices_env_var(), None)
    if cvd is None:
        return ray.get_gpu_ids()[0]
    else:
        return cvd.split(",").index(str(ray.get_gpu_ids()[0]))


class TrainRayActor(RayActor):
    def __init__(self, world_size, rank, master_addr, master_port, lock, role="actor"):
        self._world_size = world_size
        self._rank = rank
        self.lock = lock
        self.role = role
        self._reserved_master_socket = None
        if master_addr:
            self.master_addr, self.master_port = master_addr, master_port
        else:
            self.master_addr, self.master_port, self._reserved_master_socket = self._reserve_master_addr_and_port(role)

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # TODO: currently this doesn't work as ray has already set torch.cuda.device_count().
        # os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])
        os.environ["LOCAL_RANK"] = str(get_local_gpu_id())

    @staticmethod
    def _role_port_range(role):
        role_key = (role or "default").lower()
        if role_key in _ROLE_MASTER_PORT_DEFAULTS:
            default_min, default_max = _ROLE_MASTER_PORT_DEFAULTS[role_key]
            env_key = role_key.upper().replace("-", "_")
        else:
            default_min, default_max = _ROLE_MASTER_PORT_DEFAULTS["default"]
            env_key = "DEFAULT"
        port_min = int(os.environ.get(f"RELAX_TRAIN_MASTER_PORT_{env_key}_MIN", default_min))
        port_max = int(os.environ.get(f"RELAX_TRAIN_MASTER_PORT_{env_key}_MAX", default_max))
        if port_min <= 0 or port_max <= 0 or port_min > port_max or port_max > 65535:
            raise ValueError(f"Invalid train MASTER_PORT range for role={role}: {port_min}-{port_max}")
        return port_min, port_max

    @classmethod
    def _reserve_master_addr_and_port(cls, role):
        # Keep actor/reference/actor_fwd in disjoint port domains.  CP=2 creates
        # more NCCL process groups, so co-located role rank-0s can otherwise race
        # each other between "port looks free" and torch's TCPStore bind.
        port_min, port_max = cls._role_port_range(role)
        width = port_max - port_min + 1
        start_port = random.randint(port_min, port_max)
        for offset in range(width):
            port = port_min + ((start_port - port_min + offset) % width)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(("", port))
                sock.listen(1)
            except OSError:
                sock.close()
                continue
            logger.info(f"[{role}] reserved train MASTER_PORT {port} in role range {port_min}-{port_max}")
            return get_current_node_ip(), port, sock
        raise RuntimeError(f"No free train MASTER_PORT found for role={role} in range {port_min}-{port_max}")

    def _release_reserved_master_socket(self):
        if self._reserved_master_socket is None:
            return
        logger.info(f"[{self.role}] releasing reserved train MASTER_PORT {self.master_port} before init_process_group")
        self._reserved_master_socket.close()
        self._reserved_master_socket = None

    def init(self, args, role, with_ref=False, with_opd_teacher=False):
        self.args = args
        self.role = role
        self.with_ref = with_ref
        self.with_opd_teacher = with_opd_teacher

        torch.serialization.add_safe_globals([relax.utils.training.eval_config.EvalDatasetConfig])

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_utils.set_device(f"{device_utils.get_device_name()}:{local_rank}")

        backend = args.distributed_backend

        self._release_reserved_master_socket()
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )
        init_gloo_group()

        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()

        numa_local_rank = int(os.environ["RANK"]) % args.num_gpus_per_node
        device_utils.set_numa_affinity(numa_local_rank)

    def clear_memory(self):
        print_memory("before TrainRayActor.clear_memory")
        clear_memory()
        print_memory("after TrainRayActor.clear_memory")

    @abc.abstractmethod
    def sleep(self, tags):
        raise NotImplementedError

    @abc.abstractmethod
    def wake_up(self, tags):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, rollout_id, rollout_data_ref):
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, rollout_id, force_sync=False):
        raise NotImplementedError

    @abc.abstractmethod
    def update_weights(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_parallel_config(self):
        raise NotImplementedError

    def set_rollout_manager(self, rollout_manager):
        self.rollout_manager = rollout_manager
        if not self.args.debug_rollout_only and self.args.rank == 0:
            ray.get(self.rollout_manager.set_train_parallel_config.remote(self.train_parallel_config))
        # Retrieve the distributed lock that serialises DCS weight sync with
        # P2P direct sync (_sync_weights_from_seed_engine on RolloutManager).
        self._weight_sync_lock = ray.get(self.rollout_manager.get_weight_sync_lock.remote())

    def set_genrm_manager(self, genrm_manager):
        """Set the genRM manager for coordinated offload/onload.

        In colocated mode, the genRM manager is used to offload genRM engines
        before training and onload them before rollout, since they share GPU
        resources.
        """
        self.genrm_manager = genrm_manager
