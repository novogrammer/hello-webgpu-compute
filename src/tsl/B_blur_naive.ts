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
import { Fn, storage, instanceIndex, uint, int, If, mod, Loop, float, clamp } from 'three/tsl';

const ENABLE_FORCE_WEBGL = false;
const SHOW_COMPUTE_SHADER = true;

const WIDTH = 1024;
const HEIGHT = 1024;

const PIXELS = WIDTH * HEIGHT;
const WORKGROUP = 64;
const DISPATCH_X = Math.ceil(PIXELS / WORKGROUP);

const RADIUS = 3; // ← この定数を変えれば範囲可変

async function runAsync(cin: HTMLCanvasElement, cout: HTMLCanvasElement): Promise<string[]> {
  const lines: string[] = [];
  const tInit = new Timer('init');
  const tPrep = new Timer('prepare');
  const tComp = new Timer('compute');
  const tRead = new Timer('read');

  tInit.start();
  const renderer = new WebGPURenderer({ forceWebGL: ENABLE_FORCE_WEBGL });
  await renderer.init();
  tInit.stop();

  // 入力データ
  tPrep.start();
  drawCheckerBoard(cin, WIDTH, HEIGHT);
  const inF32 = toFloat32Array(getImageData(cin));

  const inputAttr = new StorageInstancedBufferAttribute(inF32, 1);
  const outputAttr = new StorageInstancedBufferAttribute(new Float32Array(inF32.length), 1);

  const inputNode = storage(inputAttr, 'float', inF32.length);
  const outputNode = storage(outputAttr, 'float', inF32.length);

  const W = uint(WIDTH);
  const H = uint(HEIGHT);
  const pixelsU = uint(PIXELS);

  const kernelFn = Fn(() => {
    const i = instanceIndex.toVar("i");
    If(i.lessThanEqual(pixelsU), () => {

      const x = int(mod(i, W)).toVar("x");
      const y = int(i.div(W)).toVar("y");

      // 出力用
      const sumR = float(0).toVar("sumR");
      const sumG = float(0).toVar("sumG");
      const sumB = float(0).toVar("sumB");
      const sumA = float(0).toVar("sumA");
      const count = float(0).toVar("count");

      // dy = -R .. +R
      Loop({start:int(-RADIUS),end:int(RADIUS),condition:'<='}, ({i}) => {
        const dy=int(i).toVar("dy");
        // dx = -R .. +R
        Loop({start:int(-RADIUS),end:int(RADIUS),condition:'<='}, ({i}) => {
          const dx=int(i).toVar("dx");
          const nx = clamp(x.add(dx),0,W).toVar("nx");
          const ny = clamp(y.add(dy),0,H).toVar("ny");
          const base = (ny.mul(W).add(nx)).mul(4).toVar("base");

          sumR.addAssign(inputNode.element(base.add(0)));
          sumG.addAssign(inputNode.element(base.add(1)));
          sumB.addAssign(inputNode.element(base.add(2)));
          sumA.addAssign(inputNode.element(base.add(3)));
          count.addAssign(1);
        });
      });

      const invCount=float(1).div(count).toVar("invCount")
      const outputBase = (y.mul(W).add(x)).mul(4).toVar("outputBase");
      outputNode.element(outputBase.add(0)).assign(sumR.mul(invCount));
      outputNode.element(outputBase.add(1)).assign(sumG.mul(invCount));
      outputNode.element(outputBase.add(2)).assign(sumB.mul(invCount));
      outputNode.element(outputBase.add(3)).assign(sumA.mul(invCount));
    });

  });

  const kernel = kernelFn().computeKernel([WORKGROUP, 1, 1]);

  tPrep.stop();

  tComp.start();
  await renderer.computeAsync(kernel, [DISPATCH_X, 1, 1]);
  tComp.stop();

  if (SHOW_COMPUTE_SHADER) {
    try {
      // @ts-ignore
      lines.push((renderer as any)._nodes.getForCompute(kernel).computeShader);
    } catch {}
  }

  tRead.start();
  const outBuffer = await renderer.getArrayBufferAsync(outputAttr);
  const outF32 = new Float32Array(outBuffer);
  tRead.stop();

  showImageData(cout, toUint8ClampedArray(outF32), WIDTH, HEIGHT);

  lines.push(`B_blur_naive (TSL)  WIDTH×HEIGHT=${WIDTH}×${HEIGHT}, R=${RADIUS}`);
  lines.push(tInit.getElapsedMessage());
  lines.push(tPrep.getElapsedMessage());
  lines.push(tComp.getElapsedMessage());
  lines.push(tRead.getElapsedMessage());

  renderer.dispose();
  return lines;
}

async function mainAsync(): Promise<void> {
  const msg = document.querySelector<HTMLTextAreaElement>('.p-demo__message');
  const btn = document.querySelector<HTMLButtonElement>('.p-demo__execute');
  const cin = document.querySelector<HTMLCanvasElement>('.p-demo__canvas--input');
  const cout = document.querySelector<HTMLCanvasElement>('.p-demo__canvas--output');
  if (!msg || !btn || !cin || !cout) throw new Error('elements not found');

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    msg.value = 'computing...';
    try {
      const warmup = await runAsync(cin, cout);
      const main = await runAsync(cin, cout);
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
