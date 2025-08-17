import '../style.scss'
import { drawCheckerBoard, getImageData, showImageData, toFloat32Array, toUint8ClampedArray } from '../utils/canvas_utils';
import Timer from '../utils/Timer';
import { disposeWebGpuAsync, initWebGpuAsync } from './wgsl_utils';

const WIDTH = 1024;
const HEIGHT = 1024;
// R を大きくすると効果が分かりやすい（例: 3, 5, 7）
const RADIUS = 3;

const WORKGROUP_X = 16;
const WORKGROUP_Y = 16;
const DISPATCH_X = Math.ceil(WIDTH / WORKGROUP_X);
const DISPATCH_Y = Math.ceil(HEIGHT / WORKGROUP_Y);

// 共有メモリ最適化（タイル＋ハロー：全域ストライド読み）
const shaderCode = /* wgsl */`
override R      : u32 = 1u;      // TS 側 constants で上書き
override TILE_X : u32 = 16u;
override TILE_Y : u32 = 16u;


struct ImageF32 { data: array<f32>, };         // RGBA 連続 (width*height*4)
struct Params   { width: u32, height: u32, _pad0: u32, _pad1: u32, }; // 16B整列

@group(0) @binding(0) var<storage, read>       inputBuf  : ImageF32;
@group(0) @binding(1) var<storage, read_write> outputBuf : ImageF32;
@group(0) @binding(2) var<uniform>             params    : Params;

var<workgroup> tile : array<vec4<f32>, (TILE_X + 2u * R) * (TILE_Y + 2u * R)>;

fn idx(x: i32, y: i32, w: i32) -> i32 { return (y * w + x) * 4; }
fn clampi(v: i32, lo: i32, hi: i32) -> i32 { return max(lo, min(v, hi)); }
fn tindex(lx: i32, ly: i32) -> i32 {
  let pad_x = i32(TILE_X + 2u * R);
  return ly * i32(pad_x) + lx;
}

@compute @workgroup_size(TILE_X, TILE_Y, 1)
fn main(
  @builtin(workgroup_id)         wid : vec3<u32>,
  @builtin(local_invocation_id)  lid : vec3<u32>,
  @builtin(global_invocation_id) gid : vec3<u32>,
) {
  let pad_x = i32(TILE_X + 2u * R);
  let pad_y = i32(TILE_Y + 2u * R);

  let w = i32(params.width);
  let h = i32(params.height);

  // このWGが担当する出力タイルの原点（グローバル座標）
  let tileOx = i32(wid.x) * i32(TILE_X);
  let tileOy = i32(wid.y) * i32(TILE_Y);

  // ---- 1) 「タイル＋ハロー」を全員で分担ロード（全域ストライド）----
  var ly = i32(lid.y);
  while (ly < i32(pad_y)) {
    var lx = i32(lid.x);
    while (lx < i32(pad_x)) {
      let gx = clampi(tileOx + lx - i32(R), 0, w - 1);
      let gy = clampi(tileOy + ly - i32(R), 0, h - 1);
      let i  = idx(gx, gy, w);
      tile[tindex(lx, ly)] = vec4<f32>(
        inputBuf.data[i+0], inputBuf.data[i+1],
        inputBuf.data[i+2], inputBuf.data[i+3]
      );
      lx += i32(TILE_X);
    }
    ly += i32(TILE_Y);
  }

  workgroupBarrier();

  // ---- 2) 共有メモリから畳み込み ----
  let gx = i32(gid.x);
  let gy = i32(gid.y);
  if (gx >= w || gy >= h) { return; } // 端の余りスレッド

  let lx0 = i32(lid.x) + i32(R);
  let ly0 = i32(lid.y) + i32(R);

  var sum = vec4<f32>(0.0);
  var count = 0;
  let Ri = i32(R);
  for (var dy = -Ri; dy <= Ri; dy = dy + 1) {
    for (var dx = -Ri; dx <= Ri; dx = dx + 1) {
      sum += tile[tindex(lx0 + dx, ly0 + dy)];
      count++;
    }
  }

  let o = idx(gx, gy, w);
  let v = sum / f32(count);
  outputBuf.data[o+0] = v.x; outputBuf.data[o+1] = v.y;
  outputBuf.data[o+2] = v.z; outputBuf.data[o+3] = v.w;
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
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module,
      entryPoint: 'main',
      // override 定数をここで注入（R/TILEはWGSL側と同期させる）
      constants: { R: RADIUS, TILE_X: WORKGROUP_X, TILE_Y: WORKGROUP_Y }
    }
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
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
  return encoder.finish();
}

async function runAsync(canvasInput: HTMLCanvasElement, canvasOutput: HTMLCanvasElement): Promise<string[]> {
  const lines: string[] = [];

  const tInit = new Timer('init');
  const tPrep = new Timer('prepare');
  const tComp = new Timer('compute');
  const tMap  = new Timer('map');

  tInit.start();
  const device = await initWebGpuAsync();
  tInit.stop();

  tPrep.start();

  // 入力画像（チェッカーボード）→ Float32 RGBA (0..1)
  drawCheckerBoard(canvasInput, WIDTH, HEIGHT);
  const inputF32 = toFloat32Array(getImageData(canvasInput));
  const byteLength = inputF32.byteLength;

  const inputBuffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(inputBuffer, 0, inputF32);

  const outputBuffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  const readBuffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  // uniform: width/height のみ（16B境界に合わせパディング2つ）
  const uniformData = new Uint32Array([WIDTH, HEIGHT, 0, 0]);
  const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  tPrep.stop();

  try {
    tComp.start();
    const cmd = makeCommandBuffer(device, uniformBuffer, inputBuffer, outputBuffer, readBuffer);
    device.queue.submit([cmd]);
    await device.queue.onSubmittedWorkDone();
    tComp.stop();

    tMap.start();
    await readBuffer.mapAsync(GPUMapMode.READ);
    const outF32 = new Float32Array(readBuffer.getMappedRange());
    showImageData(canvasOutput, toUint8ClampedArray(outF32), WIDTH, HEIGHT);
    tMap.stop();

    lines.push(`WIDTH×HEIGHT: ${WIDTH}×${HEIGHT}, R=${RADIUS} → n=${RADIUS * 2 + 1}`);
    lines.push(`in[0]: ${inputF32[0]}, out[0]: ${outF32[0]}`);
    lines.push(tInit.getElapsedMessage());
    lines.push(tPrep.getElapsedMessage());
    lines.push(tComp.getElapsedMessage());
    lines.push(tMap.getElapsedMessage());
  } finally {
    readBuffer.unmap();
    inputBuffer.destroy();
    outputBuffer.destroy();
    readBuffer.destroy();
    uniformBuffer.destroy();
    await disposeWebGpuAsync(device);
  }

  return lines;
}

async function mainAsync(): Promise<void> {
  const msg  = document.querySelector<HTMLTextAreaElement>('.p-demo__message');
  const btn  = document.querySelector<HTMLButtonElement>('.p-demo__execute');
  const cin  = document.querySelector<HTMLCanvasElement>('.p-demo__canvas--input');
  const cout = document.querySelector<HTMLCanvasElement>('.p-demo__canvas--output');
  if (!msg || !btn || !cin || !cout) throw new Error('elements not found');

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    msg.value = 'computing...';
    try {
      const warmup = await runAsync(cin, cout);
      const main   = await runAsync(cin, cout);
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
