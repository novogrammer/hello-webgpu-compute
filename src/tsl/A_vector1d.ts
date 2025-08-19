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
  timerInit.start();
  const renderer = new WebGPURenderer({
  });
  await renderer.init();
  timerInit.stop();

  const inputData = new Float32Array(NUM_ELEMENTS);
  for (let i = 0; i < NUM_ELEMENTS; i++) {
    inputData[i] = i + 1;
  }

  // GPU側のストレージバッファ属性（要: Instanced 版）
   // itemSize=1 (float)
  const inputBufferAttribute = new StorageInstancedBufferAttribute(inputData, 1);
  const outputBufferAttribute = new StorageInstancedBufferAttribute(new Float32Array(NUM_ELEMENTS), 1);
  
  const inputBufferNode = storage(inputBufferAttribute, 'float', inputData.length);
  const outputBufferNode = storage(outputBufferAttribute, 'float', inputData.length);

  // 各スレッド i について: outputBuffer[i] = inputBuffer[i] * 2
  const kernelFn = Fn(() => {
    const inputValue = inputBufferNode.element(instanceIndex);
    const outputValue = outputBufferNode.element(instanceIndex);
    outputValue.assign(inputValue.mul(2.0));
  });


  timerCompute.start();
  const kernel = kernelFn().computeKernel([WORKGROUP, 1, 1]);
  await renderer.computeAsync(kernel, [DISPATCH_X, 1, 1]);
  timerCompute.stop();

  // 結果の読み戻し
  timerRead.start();
  const outBuffer = await renderer.getArrayBufferAsync(outputBufferAttribute);

  const outputData = new Float32Array(outBuffer);
  timerRead.stop();

  lines.push(`NUM_ELEMENTS: ${NUM_ELEMENTS}`);
  lines.push(`inputData[0]: ${inputData[0]} -> outputData[0]: ${outputData[0]}`);
  lines.push(timerInit.getElapsedMessage());
  lines.push(timerCompute.getElapsedMessage());
  lines.push(timerRead.getElapsedMessage());

  // 後始末
  renderer.dispose();

  return lines;
}

async function mainAsync(): Promise<void> {
  const messageElement=document.querySelector<HTMLTextAreaElement>(".p-demo__message");
  if(!messageElement){
    throw new Error("messageElement is null");
  }

  const executeElement=document.querySelector<HTMLButtonElement>(".p-demo__execute");
  if(!executeElement){
    throw new Error("executeElement is null");
  }

  executeElement.addEventListener('click', async () => {
    executeElement.disabled = true;
    messageElement.value = 'computing...';
    try {
      // ウォームアップ + 本計測
      const warmup = await runAsync();
      const main = await runAsync();
      messageElement.value = ['ウォームアップ', ...warmup, '本計測', ...main].join('\n');
    }catch(error: any){
      alert(error?.message ?? String(error));
      console.error(error);
    } finally {
      executeElement.disabled = false;
    }
  });
}

mainAsync().catch(console.error);
