import pytorch_lightning as pl
import pynvml

class GPUMonitorCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        pynvml.nvmlInit()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._log_gpu_stats(pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._log_gpu_stats(pl_module)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._log_gpu_stats(pl_module)

    def _log_gpu_stats(self, pl_module):
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming using GPU 0
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pl_module.log('gpu_mem_used_MB', mem_info.used / 1024 / 1024)
        pl_module.log('gpu_mem_total_MB', mem_info.total / 1024 / 1024)
        pl_module.log('gpu_utilization_pct', utilization.gpu)
        pl_module.log('gpu_mem_utilization_pct', utilization.memory)
