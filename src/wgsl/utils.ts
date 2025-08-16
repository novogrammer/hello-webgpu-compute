
export async function initWebGpuAsync():Promise<GPUDevice> {
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('No GPU adapter found.');
  }
  const device = await adapter.requestDevice();
  device.lost.then(info => console.warn('GPU device lost:', info.reason, info.message));

  return device;
}