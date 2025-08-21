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

async function runAsync(cin: HTMLCanvasElement, cout: HTMLCanvasElement): Promise<string[]> {
  const lines: string[] = [];
  const tInit = new Timer('init');
  const tPrep = new Timer('prepare');
  const tComp = new Timer('compute');
  const tRead = new Timer('read');

  // --- Renderer
  tInit.start();
  const renderer = new WebGPURenderer({ forceWebGL: ENABLE_FORCE_WEBGL });
  await renderer.init();
  const isWebGPU = !!((renderer.backend as any)?.isWebGPUBackend);
  lines.push(`isWebGPUBackend: ${isWebGPU}`);
  tInit.stop();

  // --- 入力（チェッカーパターンを 0/1 として扱う）
  tPrep.start();
  drawCheckerBoard(cin, WIDTH, HEIGHT);
  const inF32 = toFloat32Array(getImageData(cin)); // RGBA(0..1)

  const inputAttr  = new StorageInstancedBufferAttribute(inF32, 4);
  const outputAttr = new StorageInstancedBufferAttribute(new Float32Array(inF32.length), 4);

  // ストレージノード化（WebGL時のみPBOをON）
  const inputNodeBase  = storage(inputAttr,  'vec4', inputAttr.count).toReadOnly().setName('input');
  const inputNode = isWebGPU ? inputNodeBase : inputNodeBase.setPBO(true);
  const outputNode = storage(outputAttr, 'vec4', outputAttr.count).setName('output');

  const W = int(WIDTH);
  const H = int(HEIGHT);
  const P = int(PIXELS);

  // --- カーネル（ナイーブ：近傍8セル合計 → ルール適用）
  // wrap（トーラス）: (x+dx+W)%W, (y+dy+H)%H
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
  const computeNode = isWebGPU
    ? kernelFn().computeKernel([WORKGROUP_X, WORKGROUP_Y, 1])
    : kernelFn().compute(PIXELS);

  tPrep.stop();

  // --- 実行
  tComp.start();
  if (isWebGPU) {
    await renderer.computeAsync(computeNode, [DISPATCH_X, DISPATCH_Y, 1]);
  } else {
    await renderer.computeAsync(computeNode); // 1D: インスタンス数 = PIXELS
  }
  tComp.stop();

  if (SHOW_COMPUTE_SHADER) {
    try {
      // @ts-ignore 内部API
      console.log((renderer as any)._nodes.getForCompute(computeNode).computeShader);
    } catch {}
  }

  // --- 読み戻し & 表示
  tRead.start();
  const outBuf = await renderer.getArrayBufferAsync(outputAttr);
  const outF32 = new Float32Array(outBuf);
  tRead.stop();

  showImageData(cout, toUint8ClampedArray(outF32), WIDTH, HEIGHT);

  lines.push(`C_game_of_life (TSL naive) WIDTH×HEIGHT=${WIDTH}×${HEIGHT}`);
  lines.push(tInit.getElapsedMessage());
  lines.push(tPrep.getElapsedMessage());
  lines.push(tComp.getElapsedMessage());
  lines.push(tRead.getElapsedMessage());

  renderer.dispose();
  return lines;
}

// ---- UI ----
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
      // 1ステップだけ。複数ステップ回したい場合はループして ping-pong してください。
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
