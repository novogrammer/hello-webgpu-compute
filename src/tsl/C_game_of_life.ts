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
import {
  Fn, storage, instanceIndex, int, float, vec4,
  If, Loop, } from 'three/tsl';

const ENABLE_FORCE_WEBGL = false;
const SHOW_COMPUTE_SHADER = true;

const WIDTH = 1024;
const HEIGHT = 1024;
const PIXELS = WIDTH * HEIGHT;

// WebGPU（2D）パラメータ
const WORKGROUP_X = 16;
const WORKGROUP_Y = 16;
const DISPATCH_X = Math.ceil(WIDTH  / WORKGROUP_X);
const DISPATCH_Y = Math.ceil(HEIGHT / WORKGROUP_Y);

async function runAsync(canvasInputElement: HTMLCanvasElement, canvasOutputElement: HTMLCanvasElement): Promise<string[]> {
  const lines: string[] = [];
  const timerInit = new Timer('init');
  const timerPrepare = new Timer('prepare');
  const timerCompute = new Timer('compute');
  const timerRead = new Timer('read');

  // --- Renderer
  timerInit.start();
  const renderer = new WebGPURenderer({
    forceWebGL: ENABLE_FORCE_WEBGL,
  });
  await renderer.init();
  const isWebGPUBackend = !!((renderer.backend as any)?.isWebGPUBackend);
  lines.push(`isWebGPUBackend: ${isWebGPUBackend}`);
  timerInit.stop();

  // --- 入力（チェッカーパターンを 0/1 として扱う）
  timerPrepare.start();
  drawCheckerBoard(canvasInputElement, WIDTH, HEIGHT);
  const inputData = toFloat32Array(getImageData(canvasInputElement)); // RGBA(0..1)

  const inputAttribute  = new StorageInstancedBufferAttribute(inputData, 4);
  const outputAttribute = new StorageInstancedBufferAttribute(new Float32Array(inputData.length), 4);

  const inputNode  = storage(inputAttribute,  'vec4', inputAttribute.count).toReadOnly().setName('input');
  if(!isWebGPUBackend){
    inputNode.setPBO(true);
  }
  const outputNode = storage(outputAttribute, 'vec4', outputAttribute.count).setName('output');

  const W = int(WIDTH);
  const H = int(HEIGHT);
  const P = int(PIXELS);

  const kernelFn = Fn(() => {
    const i = int(instanceIndex).toVar('i');
    If(i.lessThan(P), () => {
      const x = i.mod(W).toVar('x');
      const y = i.div(W).toVar('y');

      // 現在セルの状態（Rチャネルを 0/1 と解釈：checkerBoardなら既に0か1）
      const selfIdx = y.mul(W).add(x).toVar('selfIdx');
      const selfVal = inputNode.element(selfIdx).x.toVar('selfVal'); // 0 or 1

      // 近傍合計（自分は除外）
      const sum = float(0).toVar('sum');

      Loop({ start: int(-1), end: int(1), condition: '<=' }, ({ i }) => {
        const dy = int(i).toVar('dy');
        Loop({ start: int(-1), end: int(1), condition: '<=' }, ({ i }) => {
          const dx = int(i).toVar('dx');

          // if (dx==0 && dy==0) continue;
          If(dx.equal(0).and(dy.equal(0)), () => {
            // do nothing (skip self)
          });
          // wrap 近傍
          const nx = x.add(dx).add(W).mod(W).toVar('nx');
          const ny = y.add(dy).add(H).mod(H).toVar('ny');
          const nIdx = ny.mul(W).add(nx).toVar('nIdx');

          const nVal = inputNode.element(nIdx).x; // 近傍のR
          sum.addAssign(nVal);
        });
      });

      // ルール：
      //  birth = (sum == 3)
      //  survive = (sum == 2 && self == 1)
      const birth   = float(0).toVar('birth');
      const survive = float(0).toVar('survive');

      If(sum.equal(3.0), () => birth.assign(1.0));
      If(sum.equal(2.0).and(selfVal.greaterThan(0.5)), () => survive.assign(1.0));

      const next = birth.add(survive).toVar('next'); // 0 or 1
      // 表示のため RGBA 同値、αは1.0固定
      const outIdx = selfIdx;
      const outValue = vec4(next).toVar('outValue');
      outValue.w.assign(1.0);
      outputNode.element(outIdx).assign(outValue);
    });
  });

  // 実行ノード（WebGPU: 2D / WebGL: 1D）
  let computeNode;
  if(isWebGPUBackend){
    computeNode = kernelFn().computeKernel([WORKGROUP_X, WORKGROUP_Y, 1])
  }else{
    computeNode = kernelFn().compute(PIXELS);
  }
  timerPrepare.stop();

  // --- 実行
  timerCompute.start();
  if (isWebGPUBackend) {
    await renderer.computeAsync(computeNode, [DISPATCH_X, DISPATCH_Y, 1]);
  } else {
    await renderer.computeAsync(computeNode); // 1D: インスタンス数 = PIXELS
  }
  timerCompute.stop();

  if (SHOW_COMPUTE_SHADER) {
    lines.push((renderer as any)._nodes.getForCompute(computeNode).computeShader);
  }

  // --- 読み戻し & 表示
  timerRead.start();
  const outputBuffer = await renderer.getArrayBufferAsync(outputAttribute);
  const outputData = new Float32Array(outputBuffer);
  timerRead.stop();

  showImageData(canvasOutputElement, toUint8ClampedArray(outputData), WIDTH, HEIGHT);

  lines.push(`C_game_of_life (TSL naive) WIDTH×HEIGHT=${WIDTH}×${HEIGHT}`);
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
