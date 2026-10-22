"""Low-overhead macOS VM counters for exact benchmark intervals.

Uses the stable HOST_VM_INFO64 revision-1 prefix from the installed Mach SDK.
These counters are system-wide; they do not attribute swap to a process/GPU.
The instance reuses its struct and Mach host port and spawns no subprocesses.
"""
import ctypes as C
import os
import sys


class VmCounters:
    def __init__(self):
        if sys.platform != 'darwin':
            raise RuntimeError('Mach VM counters require macOS')
        class Stats(C.Structure):
            _fields_ = ([(n,C.c_uint32) for n in ('free','active','inactive','wired')]
                + [(n,C.c_uint64) for n in ('zero_fill','reactivations','pageins','pageouts','faults','cow_faults','lookups','hits','purges')]
                + [(n,C.c_uint32) for n in ('purgeable','speculative')]
                + [(n,C.c_uint64) for n in ('decompressions','compressions','swapins','swapouts')]
                + [(n,C.c_uint32) for n in ('compressor_pages','throttled','external','internal')]
                + [('uncompressed_pages',C.c_uint64)])
        assert C.sizeof(Stats)==152
        self.lib=C.CDLL('/usr/lib/libSystem.B.dylib')
        self.lib.mach_host_self.restype=C.c_uint32
        self.lib.host_statistics64.argtypes=[C.c_uint32,C.c_int,C.c_void_p,C.POINTER(C.c_uint32)]
        self.lib.host_statistics64.restype=C.c_int
        self.host=self.lib.mach_host_self()
        self.stats=Stats();self.count=C.c_uint32(38)
        self.stats_ptr=C.byref(self.stats);self.count_ptr=C.byref(self.count)
        self.page_bytes=os.sysconf('SC_PAGE_SIZE')
        self.read()

    def read(self):
        self.count.value=38
        code=self.lib.host_statistics64(self.host,4,self.stats_ptr,self.count_ptr)
        if code or self.count.value<38:
            raise RuntimeError(f'host_statistics64 failed: {code}, count={self.count.value}')
        return (self.stats.swapins,self.stats.swapouts,self.stats.decompressions,self.stats.compressions)

    def delta(self,before,after):
        values={key:(b-a)*self.page_bytes for key,a,b in zip(
            ('swap_in_bytes','swap_out_bytes','decompressed_bytes','compressed_bytes'),before,after)}
        assert all(v>=0 for v in values.values()), 'VM counters went backwards'
        return dict(values,page_bytes=self.page_bytes,scope='System-wide; sampled immediately around the measured call without subprocesses',
            swap=bool(values['swap_in_bytes'] or values['swap_out_bytes']))
