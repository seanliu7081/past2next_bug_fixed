"""Opt-in hang diagnostics: every Python process dumps all thread stacks on SIGUSR1.

ptrace (py-spy, gdb) is blocked in unprivileged containers, so put this directory on PYTHONPATH and
name an output directory; the hook is passive until signalled:

    PYTHONPATH=scripts/p2n_vla_bringup/stack_dump P2N_STACK_DIR=output/stacks \
        bash train_p2n_vla.sh --variant p2n_vla --task libero --devices 0,1 --output ...
    kill -USR1 <worker pid>          # appends that process's stacks to $P2N_STACK_DIR/stack_<pid>.txt

Without P2N_STACK_DIR the module does nothing. Forked DataLoader workers inherit the handler and
write to their parent's file.
"""
import faulthandler
import os
import signal

_directory = os.environ.get("P2N_STACK_DIR")
if _directory:
    os.makedirs(_directory, exist_ok=True)
    _file = open(os.path.join(_directory, f"stack_{os.getpid()}.txt"), "a")
    faulthandler.register(signal.SIGUSR1, file=_file, all_threads=True, chain=False)
