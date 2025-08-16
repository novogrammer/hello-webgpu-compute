import '../style.scss'

import Timer from '../Timer';
import { initWebGPUAsync } from './utils';

const WIDTH = 1024;
const HEIGHT = 1024;
// n×n カーネル：R=1 → 3×3, R=3 → 7×7 など
const RADIUS = 1;

const WORKGROUP_X = 16;
const WORKGROUP_Y = 16;
const DISPATCH_X = Math.ceil(WIDTH / WORKGROUP_X);
const DISPATCH_Y = Math.ceil(HEIGHT / WORKGROUP_Y);

const shaderCode = /* wgsl */`
struct ImageF32 { data: array<f32>, }; // RGBA連続（width*height*4）
struct Params {
  width: u32,
  height: u32,
  radius: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read>       inputBuf  : ImageF32;
@group(0) @binding(1) var<storage, read_write> outputBuf : ImageF32;
@group(0) @binding(2) var<uniform>             params    : Params;

fn idx(x: i32, y: i32, w: i32) -> i32 {
  // RGBA 4要素
  return (y * w + x) * 4;
}

fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
  return max(lo, min(v, hi));
}

@compute @workgroup_size(${WORKGROUP_X}, ${WORKGROUP_Y}, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>
) {
  let w = i32(params.width);
  let h = i32(params.height);
  let r = i32(params.radius);

  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= w || y >= h) { return; }

  var sum = vec4<f32>(0.0);
  var count = 0;

  // 近傍をナイーブに全読み（外部メモリから）
  for (var dy = -r; dy <= r; dy = dy + 1) {
    let yy = clampi(y + dy, 0, h - 1);
    for (var dx = -r; dx <= r; dx = dx + 1) {
      let xx = clampi(x + dx, 0, w - 1);
      let bi = idx(xx, yy, w);
      let v = vec4<f32>(
        inputBuf.data[bi + 0],
        inputBuf.data[bi + 1],
        inputBuf.data[bi + 2],
        inputBuf.data[bi + 3]
      );
      sum = sum + v;
      count = count + 1;
    }
  }

  let outv = sum / f32(count);
  let oi = idx(x, y, w);
  outputBuf.data[oi + 0] = outv.x;
  outputBuf.data[oi + 1] = outv.y;
  outputBuf.data[oi + 2] = outv.z;
  outputBuf.data[oi + 3] = outv.w;
}
`;

function makeCommandBuffer(
  device: GPUDevice,
  uniformBuffer: GPUBuffer,
  inputBuffer: GPUBuffer,
  outputBuffer: GPUBuffer,
  readBuffer: GPUBuffer,
) {

  const module = device.createShaderModule({ code: shaderCode });

  // bindGroupLayout（自動推論でも可。明示するほうが安定）
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ]
  });
    
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module, entryPoint: 'main' }
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
      { binding: 2, resource: { buffer: uniformBuffer } },
    ]
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y, 1);
  pass.end();

  encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputBuffer.size);

  const commandBuffer = encoder.finish();

  return commandBuffer;
}

async function runAsync(): Promise<string[]> {
  const lines: string[] = [];

  const timerInit  = new Timer('init');
  const timerPrep  = new Timer('prepare');
  const timerExec  = new Timer('compute');
  const timerRead  = new Timer('map');

  timerInit.start();
  const device = await initWebGPUAsync();
  timerInit.stop();

  timerPrep.start();

  const pixels = WIDTH * HEIGHT;
  const byteLength = pixels * 4 * 4; // RGBA * f32
  const input = new Float32Array(pixels * 4);

  // 簡単なテストパターン（横グラデ＋縦グラデ）
  for (let y = 0; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      const i = (y * WIDTH + x) * 4;
      input[i + 0] = x / WIDTH;   // R
      input[i + 1] = y / HEIGHT;  // G
      input[i + 2] = 0.25;        // B
      input[i + 3] = 1.0;         // A
    }
  }

  const inputBuffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuffer, 0, input);

  const outputBuffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  const readBuffer = device.createBuffer({
    size: outputBuffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });


  const uniformData = new Uint32Array([WIDTH, HEIGHT, RADIUS, 0]);
  const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  timerPrep.stop();

  try{

    timerExec.start();
    const commandBuffer = makeCommandBuffer(
      device, uniformBuffer, inputBuffer, outputBuffer, readBuffer
    );
    device.queue.submit([commandBuffer]);
    await device.queue.onSubmittedWorkDone();
    timerExec.stop();

    timerRead.start();
    await readBuffer.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(readBuffer.getMappedRange());
    timerRead.stop();

    lines.push(`WIDTH×HEIGHT: ${WIDTH}×${HEIGHT}, R=${RADIUS} → n=${RADIUS * 2 + 1}`);
    lines.push(`input[0]: ${input[0]}, out[0]: ${out[0]}`);
    lines.push(timerInit.getElapsedMessage());
    lines.push(timerPrep.getElapsedMessage());
    lines.push(timerExec.getElapsedMessage());
    lines.push(timerRead.getElapsedMessage());

  }finally{
    readBuffer.unmap();
    inputBuffer.destroy();
    outputBuffer.destroy();
    readBuffer.destroy();
    uniformBuffer.destroy();

    await device.queue.onSubmittedWorkDone();
    device.destroy();
    
  }


  return lines;
}

async function mainAsync(): Promise<void> {
  const msg = document.querySelector<HTMLTextAreaElement>('.p-demo__message');
  const btn = document.querySelector<HTMLButtonElement>('.p-demo__execute');
  if (!msg || !btn) throw new Error('elements not found');

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    msg.value = 'computing...';
    try {
      // ウォームアップ + 本計測
      const warmup = await runAsync();
      const main   = await runAsync();
      msg.value = ['ウォームアップ', ...warmup, '本計測', ...main].join('\n');
    } catch (e: any) {
      alert(e?.message ?? String(e));
      console.error(e);
    } finally {
      btn.disabled = false;
    }
  });
}

mainAsync().catch(console.error);
