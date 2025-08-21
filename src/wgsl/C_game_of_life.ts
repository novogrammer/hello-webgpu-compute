import '../style.scss'
import { drawCheckerBoard, getImageData, showImageData, toFloat32Array, toUint8ClampedArray } from '../utils/canvas_utils';
import Timer from '../utils/Timer';
import { disposeWebGpuAsync, initWebGpuAsync } from './wgsl_utils';

const WIDTH = 1024;
const HEIGHT = 1024;
const WORKGROUP_X = 16;
const WORKGROUP_Y = 16;
const DISPATCH_X = Math.ceil(WIDTH / WORKGROUP_X);
const DISPATCH_Y = Math.ceil(HEIGHT / WORKGROUP_Y);

// 何世代回すか
const STEPS = 200;

// === WGSL: 1世代分の更新カーネル（境界はトーラスwrap） ===
// 入出力は RGBA float(0..1) 配列だが、実際は R チャンネルだけ使います。
// R>=0.5 を「生」、R<0.5 を「死」として扱います。描画はグレースケール。
const shaderCode = /* wgsl */`
struct ImageF32 { data: array<f32>, };
struct Params   { width: u32, height: u32, _pad0: u32, _pad1: u32, }

@group(0) @binding(0) var<storage, read>       inputBuf  : ImageF32;
@group(0) @binding(1) var<storage, read_write> outputBuf : ImageF32;
@group(0) @binding(2) var<uniform>             params    : Params;

fn idx(x: i32, y: i32, w: i32) -> i32 {
  return (y * w + x) * 4;
}

// torus wrap（負値にも対応）
fn wrap(v: i32, n: i32) -> i32 {
  // ((v % n) + n) % n
  let m = v % n;
  return (m + n) % n;
}

@compute @workgroup_size(${WORKGROUP_X}, ${WORKGROUP_Y}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = i32(params.width);
  let h = i32(params.height);

  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= w || y >= h) { return; }

  // 8近傍の生存数を数える（R>=0.5 を 1 とみなす）
  var neighbors = 0;
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      if (dx == 0 && dy == 0) { continue; }
      let xx = wrap(x + dx, w);
      let yy = wrap(y + dy, h);
      let i = idx(xx, yy, w);
      let alive = select(0, 1, inputBuf.data[i] >= 0.5); // Rのみ
      neighbors = neighbors + alive;
    }
  }

  let i0 = idx(x, y, w);
  let self_alive = inputBuf.data[i0] >= 0.5;

  // Conway's Game of Life ルール
  // 生存: 生で隣人2or3 → 生, 死で隣人3 → 生, その他は死
  var next_alive = false;
  if (self_alive) {
    next_alive = (neighbors == 2) || (neighbors == 3);
  } else {
    next_alive = (neighbors == 3);
  }

  let v = select(0.0, 1.0, next_alive);
  // グレースケールで書き出し（RGBA全て同値、A=1）
  outputBuf.data[i0 + 0] = v;
  outputBuf.data[i0 + 1] = v;
  outputBuf.data[i0 + 2] = v;
  outputBuf.data[i0 + 3] = 1.0;
}
`;

function makeCommandEncoderForSteps(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroupLayout: GPUBindGroupLayout,
  uniformBuffer: GPUBuffer,
  inputA: GPUBuffer,
  inputB: GPUBuffer,
  steps: number,
  readBuffer: GPUBuffer,
) {
  // A↔B ピンポン: 偶数ステップ A→B、奇数ステップ B→A（最後に「最新」をreadback）
  let src = inputA;
  let dst = inputB;

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();

  for (let s = 0; s < steps; s++) {
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: src } },
        { binding: 1, resource: { buffer: dst } },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ]
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y, 1);

    // swap
    const tmp = src; src = dst; dst = tmp;
  }

  pass.end();

  // 最後の src が「最新」。それを readBuffer にコピー
  encoder.copyBufferToBuffer(src, 0, readBuffer, 0, readBuffer.size);
  return encoder.finish();
}

async function runAsync(canvasInput: HTMLCanvasElement, canvasOutput: HTMLCanvasElement): Promise<string[]> {
  const lines: string[] = [];
  const timerInit = new Timer('init');
  const timerPrepare = new Timer('prepare');
  const timerCompute = new Timer('compute');
  const timerMap  = new Timer('map');

  timerInit.start();
  const device = await initWebGpuAsync();
  timerInit.stop();

  timerPrepare.start();

  // 初期状態：チェッカーボードを種にする
  drawCheckerBoard(canvasInput, WIDTH, HEIGHT);
  const input = toFloat32Array(getImageData(canvasInput)); // RGBA(0..1)
  const byteLength = input.byteLength;

  // ピンポン用 2 バッファ
  const bufferA = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  });
  device.queue.writeBuffer(bufferA, 0, input.buffer);

  const bufferB = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  const readBuffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  // uniform: width/height（16B整列用のパディングを2つ）
  const uniformData = new Uint32Array([WIDTH, HEIGHT, 0, 0]);
  const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  // パイプライン
  const module = device.createShaderModule({ code: shaderCode });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' }
  });
  const bindGroupLayout = pipeline.getBindGroupLayout(0);

  timerPrepare.stop();

  try {
    timerCompute.start();
    const cmd = makeCommandEncoderForSteps(
      device, pipeline, bindGroupLayout, uniformBuffer,
      bufferA, bufferB, STEPS, readBuffer
    );
    device.queue.submit([cmd]);
    await device.queue.onSubmittedWorkDone();
    timerCompute.stop();

    timerMap.start();
    await readBuffer.mapAsync(GPUMapMode.READ);
    const output = new Float32Array(readBuffer.getMappedRange());
    // そのまま描画（1か0のグレースケール）
    showImageData(canvasOutput, toUint8ClampedArray(output), WIDTH, HEIGHT);
    timerMap.stop();

    lines.push(`Game of Life: ${WIDTH}×${HEIGHT}, steps=${STEPS}`);
    lines.push(timerInit.getElapsedMessage());
    lines.push(timerPrepare.getElapsedMessage());
    lines.push(timerCompute.getElapsedMessage());
    lines.push(timerMap.getElapsedMessage());
  } finally {
    readBuffer.unmap();
    bufferA.destroy();
    bufferB.destroy();
    readBuffer.destroy();
    uniformBuffer.destroy();
    await disposeWebGpuAsync(device);
  }

  return lines;
}

async function mainAsync(): Promise<void> {
  const messageElement = document.querySelector<HTMLTextAreaElement>('.p-demo__message');
  if(!messageElement){
    throw new Error("messageElement is null");
  }

  const executeElement = document.querySelector<HTMLButtonElement>('.p-demo__execute');
  if(!executeElement){
    throw new Error("executeElement is null");
  }

  const canvasInputElement = document.querySelector<HTMLCanvasElement>('.p-demo__canvas--input');
  if(!canvasInputElement){
    throw new Error("canvasInputElement is null");
  }
  const canvasOutputElement = document.querySelector<HTMLCanvasElement>('.p-demo__canvas--output');
  if(!canvasOutputElement){
    throw new Error("canvasOutputElement is null");
  }

  executeElement.addEventListener('click', async () => {
    executeElement.disabled = true;
    messageElement.value = 'computing...';
    try {
      // ウォームアップ + 本計測
      const warmup = await runAsync(canvasInputElement, canvasOutputElement);
      const main   = await runAsync(canvasInputElement, canvasOutputElement);
      messageElement.value = ['ウォームアップ', ...warmup, '本計測', ...main].join('\n');
    } catch (error: any) {
      alert(error?.message ?? String(error));
      console.error(error);
    } finally {
      executeElement.disabled = false;
    }
  });
}

mainAsync().catch(console.error);
