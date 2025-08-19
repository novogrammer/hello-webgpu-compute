import '../style.scss'
// src/tsl/A_vector1d.ts
import Timer from '../utils/Timer';

// three.js + TSL
import { WebGPURenderer,StorageInstancedBufferAttribute } from 'three/webgpu';
import { Fn, storage, instanceIndex } from 'three/tsl';

const NUM_ELEMENTS = 1024 * 1024;
const WORKGROUP = 64;
const DISPATCH_X = Math.ceil(NUM_ELEMENTS / WORKGROUP);

async function runAsync(): Promise<string[]> {
  const lines: string[] = [];

  const timerInit = new Timer('init');
  const timerCompute = new Timer('compute');
  const timerRead = new Timer('read');

  // renderer は compute だけに使うので描画用シーンは不要
  const renderer = new WebGPURenderer();
  timerInit.start();
  await renderer.init();
  timerInit.stop();

  // 入力データ
  const input = new Float32Array(NUM_ELEMENTS);
  for (let i = 0; i < NUM_ELEMENTS; i++) input[i] = i + 1;

  // GPU側のストレージバッファ属性（要: Instanced 版）
  const bufAttr = new StorageInstancedBufferAttribute(input, 1); // itemSize=1 (float)
  const bufNode = storage(bufAttr, 'float', input.length); // TSLノード化

  // 各スレッド i について: buf[i] = buf[i] * 2
  const kernelFn = Fn(() => {
    const v = bufNode.element(instanceIndex); // 現在のインデックス要素
    v.assign(v.mul(2.0));
  });

  // workgroupSize と dispatchSize を指定（r179以降）
  const kernel = kernelFn().computeKernel([WORKGROUP, 1, 1]);

  timerCompute.start();
  await renderer.computeAsync(kernel, [DISPATCH_X, 1, 1]);
  timerCompute.stop();

  // 結果の読み戻し
  timerRead.start();
  const outBuffer = await renderer.getArrayBufferAsync(bufAttr);
  const output = new Float32Array(outBuffer);
  timerRead.stop();

  lines.push(`NUM_ELEMENTS: ${NUM_ELEMENTS}`);
  lines.push(`input[0]: ${input[0]} -> output[0]: ${output[0]}`);
  lines.push(timerInit.getElapsedMessage());
  lines.push(timerCompute.getElapsedMessage());
  lines.push(timerRead.getElapsedMessage());

  // 後始末
  renderer.dispose();

  return lines;
}

async function mainAsync(): Promise<void> {
  const messageEl = document.querySelector<HTMLTextAreaElement>('.p-demo__message');
  if (!messageEl) throw new Error('messageElement is null');
  const execEl = document.querySelector<HTMLButtonElement>('.p-demo__execute');
  if (!execEl) throw new Error('executeElement is null');

  execEl.addEventListener('click', async () => {
    execEl.disabled = true;
    messageEl.value = 'computing...';
    try {
      const warmup = await runAsync();
      const main = await runAsync();
      messageEl.value = ['ウォームアップ', ...warmup, '本計測', ...main].join('\n');
    } catch (e: any) {
      alert(e?.message ?? String(e));
      console.error(e);
    } finally {
      execEl.disabled = false;
    }
  });
}

mainAsync().catch(console.error);
