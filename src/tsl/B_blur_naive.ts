import '../style.scss'

import Timer from '../utils/Timer';
import {
  drawCheckerBoard,
  getImageData,
  showImageData,
  toFloat32Array,
  toUint8ClampedArray,
} from '../utils/canvas_utils';

import { WebGPURenderer, StorageInstancedBufferAttribute } from 'three/webgpu';
import { Fn, instanceIndex, int, If, mod, Loop, float, clamp, vec4, storage } from 'three/tsl';

const ENABLE_FORCE_WEBGL = false;
const SHOW_COMPUTE_SHADER = false;

const WIDTH = 1024;
const HEIGHT = 1024;

const PIXELS = WIDTH * HEIGHT;

const WORKGROUP_X = 16;
const WORKGROUP_Y = 16;
const DISPATCH_X = Math.ceil(WIDTH / WORKGROUP_X);
const DISPATCH_Y = Math.ceil(HEIGHT / WORKGROUP_Y);

const RADIUS = 3; // ← この定数を変えれば範囲可変

async function runAsync(canvasInputElement: HTMLCanvasElement, canvasOutputElement: HTMLCanvasElement): Promise<string[]> {
  const lines: string[] = [];
  const timerInit = new Timer('init');
  const timerPrepare = new Timer('prepare');
  const timerCompute = new Timer('compute');
  const timerRead = new Timer('read');

  timerInit.start();
  const renderer = new WebGPURenderer({
    forceWebGL: ENABLE_FORCE_WEBGL,
  });
  await renderer.init();
  timerInit.stop();

  // 入力データ
  timerPrepare.start();
  drawCheckerBoard(canvasInputElement, WIDTH, HEIGHT);
  const inputData = toFloat32Array(getImageData(canvasInputElement));

  const inputAttribute = new StorageInstancedBufferAttribute(inputData, 4);
  const outputAttribute = new StorageInstancedBufferAttribute(new Float32Array(inputData.length), 4);

  const inputNode = storage(inputAttribute, 'vec4',inputAttribute.count).setPBO(true).toReadOnly().setName("inputNode");
  const outputNode = storage(outputAttribute, 'vec4', outputAttribute.count).setName("outputNode");

  const W = int(WIDTH).toVar("W");
  const H = int(HEIGHT).toVar("H");
  const pixels = int(PIXELS).toVar("pixels");

  const kernelFn = Fn(() => {
    const i = int(instanceIndex).toVar("i");
    If(i.lessThan(pixels), () => {

      const x = mod(i, W).toVar("x");
      const y = i.div(W).toVar("y");

      // 出力用
      const sum = vec4(0).toVar("sum");
      const count = float(0).toVar("count");

      // dy = -R .. +R
      Loop({start:int(-RADIUS),end:int(RADIUS),condition:'<='}, ({i}) => {
        const dy=int(i).toVar("dy");
        // dx = -R .. +R
        Loop({start:int(-RADIUS),end:int(RADIUS),condition:'<='}, ({i}) => {
          const dx=int(i).toVar("dx");
          const nx = clamp(x.add(dx),0,W.sub(1)).toVar("nx");
          const ny = clamp(y.add(dy),0,H.sub(1)).toVar("ny");
          const index = (ny.mul(W).add(nx)).toVar("index");

          sum.addAssign(inputNode.element(index));
          count.addAssign(1);
        });
      });

      const outputIndex = (y.mul(W).add(x)).toVar("outputIndex");
      outputNode.element(outputIndex).assign(sum.div(count));
    });

  });

  const kernel = kernelFn().computeKernel([WORKGROUP_X, WORKGROUP_Y, 1]);

  timerPrepare.stop();

  timerCompute.start();
  await renderer.computeAsync(kernel, [DISPATCH_X, DISPATCH_Y, 1]);
  timerCompute.stop();

  if (SHOW_COMPUTE_SHADER) {
    lines.push((renderer as any)._nodes.getForCompute(kernel).computeShader);
  }

  timerRead.start();
  const outputBuffer = await renderer.getArrayBufferAsync(outputAttribute);
  const outputData = new Float32Array(outputBuffer);
  timerRead.stop();

  showImageData(canvasOutputElement, toUint8ClampedArray(outputData), WIDTH, HEIGHT);

  lines.push(`B_blur_naive (TSL)  WIDTH×HEIGHT=${WIDTH}×${HEIGHT}, R=${RADIUS}`);
  lines.push(timerInit.getElapsedMessage());
  lines.push(timerPrepare.getElapsedMessage());
  lines.push(timerCompute.getElapsedMessage());
  lines.push(timerRead.getElapsedMessage());

  renderer.dispose();
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
      const main = await runAsync(canvasInputElement, canvasOutputElement);
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
